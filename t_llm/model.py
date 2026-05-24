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
# Classification Head (multi-task 전용, Section 3.5 확장)
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    메모리 시계열에서 job completion / eviction 분류.

    세 가지 정보를 결합:
      1) CNN path   : 원시 메모리 시퀀스 (B, L) → 1-D CNN → (B, CNN_DIM)
                      종료 직전 패턴(감소/급증 등)을 직접 학습
      2) stats path : 수작업 통계량 6개 (B, N_STATS)
      3) LLM path   : .detach()된 GPT-2 풀링 (B, d_model)
                      regression gradient가 classification을 방해하지 않도록 격리

    Input:
      pooled  : (B, d_model)  — GPT-2 last hidden state, mean-pooled, detached
      stats   : (B, N_STATS)  — 수작업 메모리 통계량
      mem_seq : (B, L)        — RevIN 정규화된 메모리 채널 원시 시퀀스
    Output: (B,) — raw logit (BCEWithLogitsLoss 사용)
    """

    # x_norm (RevIN 후): 패턴 피처 (trend, 후반 slope 등)
    # x_raw  (RevIN 전): 절대 수준 피처 (mean, max 등) — RevIN이 제거한 정보 복원
    #
    # CNN 대신 stats-only 소형 MLP를 사용하는 이유:
    #   instance-level split이므로 실제 독립 샘플은 job 수(~3K)에 불과하다.
    #   job당 ~70개의 겹치는 윈도우가 있어 CNN이 training job 패턴을 암기한다.
    #   → val BCE가 epoch마다 증가하는 overfitting 발생.
    #   stats 8개 + 소형 MLP + strong dropout으로 일반화 능력 확보.
    N_STATS: int = 8   # (norm: 2) + (raw: 6) — 아래 compute_stats 참고

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(self.N_STATS),   # batch 통계로 안정화 (LayerNorm 대비 일반화↑)
            nn.Linear(self.N_STATS, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        stats:   torch.Tensor,   # (B, N_STATS)
        mem_raw: torch.Tensor | None = None,  # 사용 안 함 (API 호환성 유지)
    ) -> torch.Tensor:
        return self.net(stats).squeeze(-1)   # (B,)

    @staticmethod
    def compute_stats(
        x_norm: torch.Tensor,   # (B, L, C) — RevIN 정규화 후 (패턴)
        x_raw:  torch.Tensor,   # (B, L, C) — RevIN 정규화 전 (절대 수준)
    ) -> torch.Tensor:
        """
        returns: (B, 8)
          ── x_norm 기반 (RevIN 후: 상대 패턴) ──
          [0] trend       — 정규화 시퀀스의 전체 추세 (last - first)
          [1] late_trend  — 정규화 시퀀스의 후반 추세 (last - mid)
          ── x_raw  기반 (RevIN 전: 절대 수준) ──
          [2] raw_mean    — 절대 평균 메모리 수준  ← RevIN이 제거한 핵심 피처
          [3] raw_std     — 절대 변동성
          [4] raw_max     — 절대 최대 메모리 (자원 압박)
          [5] raw_last    — 마지막 관측 메모리 (종료 직전 수준)
          [6] raw_range   — 절대 변동 폭
          [7] raw_late_tr — 절대값 기준 후반 추세 (last - mid)
        """
        mem_n = x_norm[:, :, 0]   # (B, L) — RevIN 후 패턴
        mem_r = x_raw[:,  :, 0]   # (B, L) — RevIN 전 절대 수준
        half = mem_n.shape[1] // 2

        # 패턴 피처 (RevIN 정규화 후 — 상대 기울기)
        trend      = mem_n[:, -1] - mem_n[:, 0]
        late_trend = mem_n[:, -1] - mem_n[:, half]

        # 절대 수준 피처 (RevIN 정규화 전 — 중요한 discriminative 신호)
        raw_mean    = mem_r.mean(dim=1)
        raw_std     = mem_r.std(dim=1).clamp(min=1e-6)
        raw_max     = mem_r.max(dim=1).values
        raw_last    = mem_r[:, -1]
        raw_range   = raw_max - mem_r.min(dim=1).values
        raw_late_tr = mem_r[:, -1] - mem_r[:, half]

        return torch.stack(
            [trend, late_trend, raw_mean, raw_std, raw_max, raw_last, raw_range, raw_late_tr],
            dim=1,
        )


# ---------------------------------------------------------------------------
# T-LLM  (Section 3.2)
# ---------------------------------------------------------------------------

class TLLM(nn.Module):
    """
    Full T-LLM model combining InputBlock, TemporalTeacher, and GPT2LoRAStudent.

    Training:  forward(x, teacher=True)  → returns teacher + student outputs
    Inference: predict(x)                → student prediction only (ch0 memory)
               predict_cls(x)            → cls_logit (use_cls_head=True 시)
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

        # 분류 head (use_cls_head=True 일 때만 생성)
        self.cls_head = ClassificationHead(cfg.dropout) if cfg.use_cls_head else None

    def _cls_logit(
        self,
        x_norm: torch.Tensor,   # RevIN 정규화 후 — 패턴 피처용
        x_raw:  torch.Tensor,   # RevIN 정규화 전 — 절대 수준 피처 + CNN 입력
    ) -> torch.Tensor:
        """
        CNN(x_raw) + stats(x_norm, x_raw) → logit.

        LLM pooled 피처를 사용하지 않는다:
          LoRA가 regression으로 계속 학습되면 pooled 값이 매 step 변해
          cls_head가 moving target을 쫓아 진동한다.
          CNN + raw stats만 쓰면 LoRA와 완전히 독립적으로 안정 수렴.
        """
        stats   = ClassificationHead.compute_stats(x_norm, x_raw)  # (B, 8)
        mem_raw = x_raw[:, :, 0]                                    # (B, L) RevIN 이전 memory
        return self.cls_head(stats, mem_raw)                        # (B,)

    def forward(self, x: torch.Tensor, teacher: bool = True) -> dict[str, object]:
        B = x.size(0)
        x_norm, mean, std = self.revin.normalize(x)
        teacher_tokens, student_tokens = self.input(x_norm)
        s_pred_norm, s_feat = self.student(student_tokens)
        s_pred = self.revin.denormalize(s_pred_norm, mean, std)
        result: dict[str, object] = {"student_pred": s_pred, "student_features": s_feat}

        if self.cls_head is not None:
            result["cls_logit"] = self._cls_logit(x_norm, x)  # (B,)  x=RevIN 이전

        if teacher:
            t_pred_norm, t_feat = self.teacher(teacher_tokens)
            t_pred = self.revin.denormalize(t_pred_norm, mean, std)
            result["teacher_pred"]     = t_pred
            result["teacher_features"] = t_feat
        return result

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """메모리 예측 (ch0). ETTh1 및 Trace 공용."""
        self.eval()
        x_norm, mean, std = self.revin.normalize(x)
        _, student_tokens = self.input(x_norm)
        pred_norm, _ = self.student(student_tokens)
        return self.revin.denormalize(pred_norm, mean, std)

    @torch.no_grad()
    def predict_cls(self, x: torch.Tensor) -> torch.Tensor:
        """분류 logit 반환 (use_cls_head=True 전용). shape: (B,)"""
        assert self.cls_head is not None, "use_cls_head=False 로 생성된 모델"
        self.eval()
        x_norm, _, _ = self.revin.normalize(x)
        return self._cls_logit(x_norm, x)  # LLM forward 불필요 — CNN+stats만 사용

    @torch.no_grad()
    def predict_teacher(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_norm, mean, std = self.revin.normalize(x)
        teacher_tokens, _ = self.input(x_norm)
        t_pred_norm, _ = self.teacher(teacher_tokens)
        return self.revin.denormalize(t_pred_norm, mean, std)
