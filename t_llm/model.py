import math

import torch
import torch.nn as nn
from transformers import GPT2Model

from .config import TLLMConfig
from .layers import (
    AdaptiveSpectralBlock,
    DLinearTrendBlock,
    ForecastHead,
    InputBlock,
    RevIN,
    TrendPeriodicFusion,
)


# ---------------------------------------------------------------------------
# LoRA wrapper for GPT-2 Conv1D
# ---------------------------------------------------------------------------

class LoRAConv1D(nn.Module):
    """Low-rank adaptation of a single GPT-2 Conv1D layer."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base  = base
        self.scale = alpha / max(rank, 1)
        in_dim, out_dim = base.weight.shape  # Conv1D stores weight as (in, out)
        self.dropout = nn.Dropout(dropout)
        self.lora_a  = nn.Parameter(torch.empty(in_dim, rank))
        self.lora_b  = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.dropout(x).matmul(self.lora_a).matmul(self.lora_b) * self.scale


def _wrap_lora(block: nn.Module, rank: int, alpha: float, dropout: float) -> None:
    """Replace c_attn, c_proj, c_fc, c_proj in one GPT-2 block with LoRA variants."""
    for path in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
        parts  = path.split(".")
        parent = block
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRAConv1D(child, rank, alpha, dropout))


# ---------------------------------------------------------------------------
# Temporal Teacher Branch (Section 3.4)
# ---------------------------------------------------------------------------

class TemporalTeacher(nn.Module):
    """
    N=2 stacked Temp-Spec blocks, each consisting of:
      DLinearTrendBlock  (trend modelling, Section 3.4.1)
      AdaptiveSpectralBlock (frequency modelling, Section 3.4.2)
      TrendPeriodicFusion   (horizon-aware gate, Section 3.4.3)
    Followed by a ForecastHead.
    """

    def __init__(self, cfg: TLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trend_blocks = nn.ModuleList([
            DLinearTrendBlock(cfg.d_model, cfg.moving_average_kernel)
            for _ in range(cfg.teacher_layers)
        ])
        self.spectral_blocks = nn.ModuleList([
            AdaptiveSpectralBlock(cfg.channels, cfg.d_model, cfg.spectral_bins, cfg.dropout)
            for _ in range(cfg.teacher_layers)
        ])
        self.fusion = TrendPeriodicFusion(cfg.d_model)
        self.head   = ForecastHead(cfg.d_model, cfg.prediction_length)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = tokens
        features: dict[str, torch.Tensor] = {}
        for i, (tb, sb) in enumerate(zip(self.trend_blocks, self.spectral_blocks)):
            trend    = tb(h)
            periodic = sb(h)
            h = self.fusion(trend, periodic, self.cfg.prediction_length, self.cfg.context_length)
            if i == 0:
                features["early"] = h
        features["late"] = h
        return self.head(h), features


# ---------------------------------------------------------------------------
# GPT-2 Student Branch (Section 3.5)
# ---------------------------------------------------------------------------

class GPT2LoRAStudent(nn.Module):
    """
    First cfg.student_layers (default 6) blocks of GPT-2 with LoRA adapters.
    Backbone weights are frozen; only LoRA parameters are trained.
    """

    def __init__(self, cfg: TLLMConfig) -> None:
        super().__init__()
        gpt2 = GPT2Model.from_pretrained(cfg.gpt2_model_name)
        # keep only the first student_layers transformer blocks
        gpt2.h = nn.ModuleList(list(gpt2.h[: cfg.student_layers]))
        gpt2.config.n_layer = len(gpt2.h)

        # freeze all backbone parameters
        for p in gpt2.parameters():
            p.requires_grad = False

        # inject LoRA into every retained block
        for block in gpt2.h:
            _wrap_lora(block, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)

        self.gpt2 = gpt2
        self.head = ForecastHead(cfg.d_model, cfg.prediction_length)

        # store word embeddings so InputBlock can build the compact dictionary
        self._word_emb = gpt2.wte.weight  # (vocab, d_model)

    @property
    def word_embeddings(self) -> torch.Tensor:
        return self._word_emb

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.gpt2(inputs_embeds=tokens, output_hidden_states=True,
                        use_cache=False, return_dict=True)
        hidden_states = out.hidden_states   # tuple: embedding + N layer outputs
        features = {
            # "early": output after 2nd transformer block (1-indexed layer 2)
            "early": hidden_states[2] if len(hidden_states) > 2 else out.last_hidden_state,
            # "late": output after the final transformer block
            "late":  out.last_hidden_state,
        }
        return self.head(out.last_hidden_state), features


# ---------------------------------------------------------------------------
# T-LLM  (Section 3.2)
# ---------------------------------------------------------------------------

class TLLM(nn.Module):
    """
    Full T-LLM model combining InputBlock, TemporalTeacher, and GPT2LoRAStudent.

    Training:  forward(x, teacher=True)  → returns teacher + student outputs
    Inference: predict(x)                → student prediction only
    """

    def __init__(self, cfg: TLLMConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.revin   = RevIN(cfg.channels)
        self.student = GPT2LoRAStudent(cfg)
        self.input   = InputBlock(
            context_length  = cfg.context_length,
            d_model         = cfg.d_model,
            n_heads         = cfg.n_heads,
            dropout         = cfg.dropout,
            word_embeddings = self.student.word_embeddings,
            dictionary_size = cfg.dictionary_size,
        )
        self.teacher = TemporalTeacher(cfg)

    def forward(self, x: torch.Tensor, teacher: bool = True) -> dict[str, object]:
        x_norm, mean, std = self.revin.normalize(x)
        teacher_tokens, student_tokens = self.input(x_norm)
        s_pred_norm, s_feat = self.student(student_tokens)
        s_pred = self.revin.denormalize(s_pred_norm, mean, std)
        result: dict[str, object] = {"student_pred": s_pred, "student_features": s_feat}
        if teacher:
            t_pred_norm, t_feat = self.teacher(teacher_tokens)
            t_pred = self.revin.denormalize(t_pred_norm, mean, std)
            result["teacher_pred"]     = t_pred
            result["teacher_features"] = t_feat
        return result

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_norm, mean, std = self.revin.normalize(x)
        _, student_tokens = self.input(x_norm)
        pred_norm, _ = self.student(student_tokens)
        return self.revin.denormalize(pred_norm, mean, std)

    @torch.no_grad()
    def predict_teacher(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_norm, mean, std = self.revin.normalize(x)
        teacher_tokens, _ = self.input(x_norm)
        t_pred_norm, _ = self.teacher(teacher_tokens)
        return self.revin.denormalize(t_pred_norm, mean, std)
