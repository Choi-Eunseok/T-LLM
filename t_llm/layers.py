import math

import torch
from torch import nn
from torch.nn import functional as F


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = (self.kernel_size - 1) // 2
        batch, channels, features = x.shape
        trend = F.avg_pool1d(
            x.reshape(batch * channels, 1, features),
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        ).reshape(batch, channels, features)
        seasonal = x - trend
        return trend, seasonal


class InputBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        llm_dictionary: torch.Tensor | None = None,
        dictionary_size: int = 1024,
    ) -> None:
        super().__init__()
        self.channel_embedding = nn.Linear(context_length, d_model)
        self.channel_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.teacher_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.student_norm = nn.LayerNorm(d_model)

        if llm_dictionary is None:
            self.learned_dictionary = nn.Parameter(torch.randn(dictionary_size, d_model) * 0.02)
            self.register_buffer("llm_dictionary", torch.empty(0), persistent=False)
        else:
            dictionary = self._compact_dictionary(llm_dictionary.detach(), dictionary_size)
            self.learned_dictionary = None
            self.register_buffer("llm_dictionary", dictionary, persistent=False)

    @staticmethod
    def _compact_dictionary(embedding: torch.Tensor, dictionary_size: int) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError("llm_dictionary must have shape [vocab_size, d_model]")
        size = min(dictionary_size, embedding.size(0))
        indices = torch.linspace(0, embedding.size(0) - 1, steps=size, device=embedding.device).round().long()
        return embedding.index_select(0, indices).clone()

    def dictionary(self) -> torch.Tensor:
        if self.learned_dictionary is not None:
            return self.learned_dictionary
        return self.llm_dictionary

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Treat channels as tokens and history as the token feature vector.
        tokens = self.channel_embedding(x.transpose(1, 2))
        attended, _ = self.channel_attention(tokens, tokens, tokens, need_weights=False)
        teacher_tokens = self.teacher_norm(attended)

        dictionary = self.dictionary().to(dtype=teacher_tokens.dtype, device=teacher_tokens.device)
        dictionary = dictionary.unsqueeze(0).expand(teacher_tokens.size(0), -1, -1)
        student_tokens, _ = self.cross_attention(teacher_tokens, dictionary, dictionary, need_weights=False)
        student_tokens = self.student_norm(student_tokens)
        return teacher_tokens, student_tokens


class DLinearTrendBlock(nn.Module):
    def __init__(self, channels: int, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.decomposition = MovingAverageDecomposition(kernel_size)
        self.trend_linear = nn.Linear(d_model, d_model)
        self.seasonal_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        trend, seasonal = self.decomposition(tokens)
        out = self.trend_linear(trend) + self.seasonal_linear(seasonal)
        return self.norm(out + tokens)


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, channels: int, d_model: int, spectral_bins: int, dropout: float) -> None:
        super().__init__()
        self.full_bins = d_model // 2 + 1
        self.spectral_bins = max(1, min(spectral_bins, self.full_bins))
        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.dsp_projection = nn.Parameter(torch.empty(self.full_bins, self.spectral_bins))
        self.global_real = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.global_imag = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.local_real = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.local_imag = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.frequency_embedding = nn.Parameter(torch.randn(self.spectral_bins, d_model) * 0.02)
        self.pool_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.dsp_projection)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(tokens, dim=-1)
        spectrum = torch.einsum("bcf,fk->bck", spectrum, self.dsp_projection.to(spectrum.dtype))

        power = spectrum.abs().pow(2)
        mask = (power > self.threshold).to(spectrum.dtype)

        global_weight = torch.complex(self.global_real, self.global_imag)
        local_weight = torch.complex(self.local_real, self.local_imag)
        filtered = spectrum * global_weight.unsqueeze(0) + spectrum * mask * local_weight.unsqueeze(0)
        freq_tokens = filtered.real.unsqueeze(-1) * self.frequency_embedding.unsqueeze(0).unsqueeze(0)

        weights = torch.softmax(self.pool_score(freq_tokens).squeeze(-1), dim=-1)
        pooled = torch.sum(freq_tokens * weights.unsqueeze(-1), dim=2)
        return self.norm(tokens + pooled)


class TrendPeriodicFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, trend: torch.Tensor, periodic: torch.Tensor, prediction_length: int, context_length: int) -> torch.Tensor:
        horizon = torch.full(
            (trend.size(0), trend.size(1), 1),
            prediction_length / max(context_length, 1),
            dtype=trend.dtype,
            device=trend.device,
        )
        trend_summary = trend.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.gate(torch.cat([trend_summary, horizon], dim=-1)))
        return gate * periodic + (1.0 - gate) * trend


class ForecastHead(nn.Module):
    def __init__(self, channels: int, d_model: int, prediction_length: int) -> None:
        super().__init__()
        self.channels = channels
        self.prediction_length = prediction_length
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, prediction_length),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        forecast = self.proj(tokens)
        return forecast.transpose(1, 2)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_tokens: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(max_tokens).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_tokens, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.encoding[:, : tokens.size(1)].to(tokens.dtype)
