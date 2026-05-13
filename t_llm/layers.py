import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Moving-average decomposition (DLinear)
# ---------------------------------------------------------------------------

class MovingAvgDecomp(nn.Module):
    """Separate a sequence into trend and seasonal components."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        # keep kernel odd so padding is symmetric
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.pad = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, d_model)
        B, C, D = x.shape
        flat = x.reshape(B * C, 1, D)
        trend = F.avg_pool1d(flat, self.kernel_size, stride=1, padding=self.pad,
                             count_include_pad=False).reshape(B, C, D)
        seasonal = x - trend
        return trend, seasonal


# ---------------------------------------------------------------------------
# DLinear trend block (Section 3.4.1)
# ---------------------------------------------------------------------------

class DLinearTrendBlock(nn.Module):
    """
    Decompose into trend + seasonal, project each with a separate linear,
    sum, and apply a residual + LayerNorm.
    """

    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.decomp = MovingAvgDecomp(kernel_size)
        self.trend_proj   = nn.Linear(d_model, d_model)
        self.season_proj  = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, seasonal = self.decomp(x)
        out = self.trend_proj(trend) + self.season_proj(seasonal)
        return self.norm(out + x)


# ---------------------------------------------------------------------------
# Adaptive Spectral Block (Section 3.4.2 / Appendix A.1-A.2)
# ---------------------------------------------------------------------------

class AdaptiveSpectralBlock(nn.Module):
    """
    FFT → learnable threshold mask → global + local modulation →
    Dominant Spectral Projection (DSP) → frequency-token lifting →
    attention-weighted pooling → residual + LayerNorm.
    """

    def __init__(self, channels: int, d_model: int, spectral_bins: int, dropout: float) -> None:
        super().__init__()
        self.full_bins    = d_model // 2 + 1
        self.spectral_bins = max(1, min(spectral_bins, self.full_bins))

        # learnable threshold θ (Eq. 26)
        self.threshold = nn.Parameter(torch.tensor(0.1))

        # DSP projection: d_FFT → d_red (Eq. 12)
        self.dsp = nn.Parameter(torch.empty(self.full_bins, self.spectral_bins))
        nn.init.xavier_uniform_(self.dsp)

        # global and local spectral modulation coefficients Γ_G, Γ_L (Eq. 27)
        self.gamma_g_real = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.gamma_g_imag = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.gamma_l_real = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)
        self.gamma_l_imag = nn.Parameter(torch.randn(channels, self.spectral_bins) * 0.02)

        # frequency token projection W_f: 1 → d_model (Appendix A.2, Eq. 29)
        self.freq_proj = nn.Parameter(torch.randn(self.spectral_bins, d_model) * 0.02)

        # two-layer MLP for attention weights α (Eq. 30)
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, d_model)
        # --- FFT (Eq. 25) ---
        spectrum = torch.fft.rfft(x, dim=-1)          # (B, C, d_FFT)

        # --- DSP (Eq. 12) ---
        spectrum = torch.einsum("bcf,fk->bck", spectrum, self.dsp.to(spectrum.dtype))

        # --- adaptive mask (Eq. 26) ---
        power = spectrum.abs().pow(2)
        mask  = (power > self.threshold).to(spectrum.dtype)

        # --- global + local modulation (Eq. 27-28) ---
        gamma_g = torch.complex(self.gamma_g_real, self.gamma_g_imag)   # (C, bins)
        gamma_l = torch.complex(self.gamma_l_real, self.gamma_l_imag)
        f_g = spectrum * gamma_g.unsqueeze(0)
        f_l = spectrum * mask * gamma_l.unsqueeze(0)
        fspec = f_g + f_l                                               # (B, C, bins)

        # --- lift to d_model space (Eq. 29): z_spec (B, C, bins, d_model) ---
        z = fspec.real.unsqueeze(-1) * self.freq_proj.unsqueeze(0).unsqueeze(0)

        # --- attention pooling (Eq. 30-31) ---
        alpha  = torch.softmax(self.attn_mlp(z).squeeze(-1), dim=-1)    # (B, C, bins)
        pooled = (z * alpha.unsqueeze(-1)).sum(dim=2)                   # (B, C, d_model)

        return self.norm(x + pooled)


# ---------------------------------------------------------------------------
# Trend–Periodic Fusion Gate (Section 3.4.3 / Appendix A.3)
# ---------------------------------------------------------------------------

class TrendPeriodicFusion(nn.Module):
    """
    Horizon-aware gate: E2 = g ⊙ F̃ + (1−g) ⊙ H̃  (Eq. 32-33)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # g_in = [mean(H̃), T/L] → shape (..., 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        trend: torch.Tensor,
        periodic: torch.Tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.Tensor:
        # trend, periodic: (B, C, d_model)
        horizon_ratio = prediction_length / max(context_length, 1)
        trend_mean = trend.mean(dim=-1, keepdim=True)                    # (B, C, 1)
        h = torch.full_like(trend_mean, horizon_ratio)
        g_in = torch.cat([trend_mean, h], dim=-1)                        # (B, C, 2)
        g = torch.sigmoid(self.gate_mlp(g_in))                          # (B, C, 1)
        return g * periodic + (1.0 - g) * trend


# ---------------------------------------------------------------------------
# Input Block (Section 3.3)
# ---------------------------------------------------------------------------

class InputBlock(nn.Module):
    """
    X (B, L, C)
      → channel embedding E0 (B, C, d)
      → channel self-attention E1 (B, C, d)           → teacher tokens
      → cross-attention with compact word dict Z1      → student tokens
    """

    def __init__(
        self,
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        word_embeddings: torch.Tensor,
        dictionary_size: int = 1024,
    ) -> None:
        super().__init__()
        vocab_size = word_embeddings.size(0)

        # Eq. 3: embed each channel's history (length L) into d_model
        self.channel_embed = nn.Linear(context_length, d_model)

        # Eq. 4-5: channel self-attention
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.teacher_norm = nn.LayerNorm(d_model)

        # Compact dictionary D_hat: project full vocab to P tokens (Eq. 6)
        self.vocab_proj = nn.Linear(vocab_size, dictionary_size, bias=False)
        nn.init.xavier_uniform_(self.vocab_proj.weight)
        self.register_buffer("source_vocab", word_embeddings.detach().clone(), persistent=False)

        # Eq. 6-7: cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.student_norm = nn.LayerNorm(d_model)

    def _compact_dict(self) -> torch.Tensor:
        # source_vocab: (vocab, d_model) → transpose to (d_model, vocab) → proj → (d_model, P) → T
        return self.vocab_proj(
            self.source_vocab.to(self.vocab_proj.weight.dtype).T
        ).T  # (P, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C)
        # treat channels as tokens: (B, C, L) → embed → (B, C, d)
        tokens = self.channel_embed(x.transpose(1, 2))

        # channel self-attention
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        e1 = self.teacher_norm(tokens + attn_out)               # teacher tokens

        # cross-attention with compact word dictionary
        d_hat = self._compact_dict()                             # (P, d_model)
        d_hat = d_hat.unsqueeze(0).expand(e1.size(0), -1, -1)  # (B, P, d_model)
        cross_out, _ = self.cross_attn(e1, d_hat, d_hat, need_weights=False)
        z1 = self.student_norm(e1 + cross_out)                  # student tokens

        return e1, z1


# ---------------------------------------------------------------------------
# Forecast head (Eq. 20)
# ---------------------------------------------------------------------------

class ForecastHead(nn.Module):
    """MLP head: (B, C, d) → (B, T, C)"""

    def __init__(self, d_model: int, prediction_length: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, prediction_length),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, C, d) → proj: (B, C, T) → (B, T, C)
        return self.proj(tokens).transpose(1, 2)
