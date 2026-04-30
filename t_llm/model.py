import copy
import math

import torch
from torch import nn

from .config import TLLMConfig
from .layers import (
    AdaptiveSpectralBlock,
    DLinearTrendBlock,
    ForecastHead,
    InputBlock,
    SinusoidalPositionEncoding,
    TrendPeriodicFusion,
)


class LoRAConv1D(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        in_features, out_features = base.weight.shape
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update = self.dropout(x).matmul(self.lora_a).matmul(self.lora_b) * self.scale
        return base_out + update


def _set_child_module(parent: nn.Module, name: str, child: nn.Module) -> None:
    parts = name.split(".")
    module = parent
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], child)


def _apply_lora_to_gpt2_blocks(blocks: nn.ModuleList, rank: int, alpha: float, dropout: float) -> None:
    targets = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    for block in blocks:
        for target in targets:
            module = block
            for part in target.split("."):
                module = getattr(module, part)
            _set_child_module(block, target, LoRAConv1D(module, rank, alpha, dropout))


class TemporalTeacher(nn.Module):
    def __init__(self, config: TLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.trend_blocks = nn.ModuleList(
            [
                DLinearTrendBlock(config.channels, config.d_model, config.moving_average_kernel)
                for _ in range(config.teacher_layers)
            ]
        )
        self.spectral_blocks = nn.ModuleList(
            [
                AdaptiveSpectralBlock(config.channels, config.d_model, config.spectral_bins, config.dropout)
                for _ in range(config.teacher_layers)
            ]
        )
        self.fusion = TrendPeriodicFusion(config.d_model)
        self.head = ForecastHead(config.channels, config.d_model, config.prediction_length)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trend = tokens
        periodic = tokens
        features: dict[str, torch.Tensor] = {}

        for i, (trend_block, spectral_block) in enumerate(zip(self.trend_blocks, self.spectral_blocks)):
            trend = trend_block(trend)
            periodic = spectral_block(periodic)
            fused = self.fusion(trend, periodic, self.config.prediction_length, self.config.context_length)
            if i == 0:
                features["head"] = fused
            trend = fused
            periodic = fused

        features["tail"] = fused
        return self.head(fused), features


class TransformerStudent(nn.Module):
    def __init__(self, config: TLLMConfig) -> None:
        super().__init__()
        self.position = SinusoidalPositionEncoding(config.channels, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config.student_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = ForecastHead(config.channels, config.d_model, config.prediction_length)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.position(tokens)
        features: dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i == 0:
                features["head"] = hidden
        hidden = self.final_norm(hidden)
        features["tail"] = hidden
        return self.head(hidden), features


class GPT2LoRAStudent(nn.Module):
    def __init__(self, config: TLLMConfig) -> None:
        super().__init__()
        try:
            from transformers import GPT2Model
        except ImportError as exc:
            raise ImportError("GPT2LoRAStudent requires `pip install transformers`.") from exc

        self.gpt2 = GPT2Model.from_pretrained(config.gpt2_model_name)
        self.gpt2.h = nn.ModuleList(list(self.gpt2.h[: config.student_layers]))
        hidden_size = self.gpt2.config.n_embd
        if config.d_model != hidden_size:
            raise ValueError(
                f"GPT-2 hidden size is {hidden_size}, but config.d_model={config.d_model}. "
                "Use --student-type gpt2_lora --d-model 768 for gpt2."
            )

        self.gpt2.config.n_layer = len(self.gpt2.h)
        for parameter in self.gpt2.parameters():
            parameter.requires_grad = False
        _apply_lora_to_gpt2_blocks(
            self.gpt2.h,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.head = ForecastHead(config.channels, config.d_model, config.prediction_length)

    def word_embeddings(self) -> torch.Tensor:
        return self.gpt2.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.gpt2(inputs_embeds=tokens, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        features = {
            "head": hidden_states[1] if len(hidden_states) > 1 else outputs.last_hidden_state,
            "tail": outputs.last_hidden_state,
        }
        return self.head(outputs.last_hidden_state), features


class TLLM(nn.Module):
    def __init__(self, config: TLLMConfig) -> None:
        super().__init__()
        self.config = config
        if config.student_type == "transformer":
            self.student = TransformerStudent(config)
            llm_dictionary = None
        elif config.student_type == "gpt2_lora":
            self.student = GPT2LoRAStudent(config)
            llm_dictionary = self.student.word_embeddings()
        else:
            raise ValueError(f"Unknown student_type: {config.student_type}")
        self.input_block = InputBlock(
            config.channels,
            config.context_length,
            config.d_model,
            config.n_heads,
            config.dropout,
            llm_dictionary=llm_dictionary,
            dictionary_size=config.llm_dictionary_size,
        )
        self.teacher = TemporalTeacher(config)

    def forward(self, x: torch.Tensor, include_teacher: bool = True) -> dict[str, object]:
        teacher_tokens, student_tokens = self.input_block(x)
        student_pred, student_features = self.student(student_tokens)
        output: dict[str, object] = {
            "student_pred": student_pred,
            "student_features": student_features,
        }
        if include_teacher:
            teacher_pred, teacher_features = self.teacher(teacher_tokens)
            output["teacher_pred"] = teacher_pred
            output["teacher_features"] = teacher_features
        return output

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, include_teacher=False)["student_pred"]
