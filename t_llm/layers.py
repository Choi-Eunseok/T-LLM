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
        trend = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        ).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal


class InputBlock(nn.Module):
    def __init__(self, channels: int, context_length: int, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.channel_embedding = nn.Linear(context_length, d_model)
        self.channel_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat channels as tokens and history as the token feature vector.
        tokens = self.channel_embedding(x.transpose(1, 2))
        attended, _ = self.channel_attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attended)


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
        self.spectral_bins = max(1, min(spectral_bins, d_model // 2 + 1))
        self.threshold = nn.Parameter(torch.tensor(0.1))
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(tokens, dim=-1)
        spectrum = spectrum[:, :, : self.spectral_bins]
        if spectrum.size(-1) < self.spectral_bins:
            pad = self.spectral_bins - spectrum.size(-1)
            spectrum = F.pad(spectrum, (0, pad))

        power = spectrum.abs().pow(2)
        cutoff = power.detach().mean(dim=-1, keepdim=True) + self.threshold.sigmoid() * power.detach().std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )
        mask = (power >= cutoff).to(spectrum.dtype)

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
            nn.Linear(d_model * 2 + 1, d_model),
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
        alpha = torch.sigmoid(self.gate(torch.cat([trend, periodic, horizon], dim=-1)))
        return alpha * trend + (1.0 - alpha) * periodic


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
