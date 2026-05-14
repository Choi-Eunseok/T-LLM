"""
Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
Jin et al. (2024), ICLR 2024 — https://arxiv.org/abs/2310.01728

Architecture (GPT-2 backbone):
  1. RevIN  — per-sample channel-wise normalization
  2. Patching — sliding window → (B*C, num_patches, patch_len)
  3. Patch Embedding — Linear(patch_len → d_model)
  4. Reprogramming — cross-attention: Q=patches, K/V=word prototypes → d_llm space
  5. Frozen GPT-2 — processes reprogrammed tokens
  6. Output Projection — flatten + Linear → (B, T, C)
  7. RevIN denormalize

Only the patch embedding, reprogramming layer, and output projection are trained.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TimeLLMConfig:
    context_length:    int   = 96
    prediction_length: int   = 96
    channels:          int   = 7

    patch_len:         int   = 16
    stride:            int   = 8

    d_model:           int   = 32    # patch embedding dimension
    d_llm:             int   = 768   # GPT-2 hidden size
    n_heads:           int   = 8     # reprogramming attention heads
    dropout:           float = 0.1

    n_text_prototypes: int   = 1000  # word embedding prototypes for K/V
    gpt2_model_name:   str   = "gpt2"

    @property
    def num_patches(self) -> int:
        # number of patches from a context of length context_length
        return (self.context_length - self.patch_len) // self.stride + 1


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta  = nn.Parameter(torch.zeros(channels))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta, mean, std

    def denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - self.beta) / (self.gamma + self.eps) * (std + self.eps) + mean


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Slice each channel's history into overlapping patches and project to d_model.
    Input : (B, L, C)
    Output: (B*C, num_patches, d_model)
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        # treat each channel independently: (B, L, C) → (B*C, L)
        x = x.transpose(1, 2).reshape(B * C, L)
        # unfold into patches: (B*C, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # project each patch to d_model
        return self.drop(self.norm(self.proj(x)))


# ---------------------------------------------------------------------------
# Reprogramming Layer
# ---------------------------------------------------------------------------

class ReprogrammingLayer(nn.Module):
    """
    Multi-head cross-attention that maps patch embeddings (d_model) into
    the LLM's representation space (d_llm) by attending over word prototypes.

    Q: patch embeddings  (B*C, num_patches, d_model) → projected to d_llm
    K, V: word prototypes (n_proto, d_llm)            → projected within d_llm
    Output: (B*C, num_patches, d_llm)
    """

    def __init__(self, d_model: int, n_heads: int, d_llm: int, dropout: float) -> None:
        super().__init__()
        assert d_llm % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_llm // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_llm, bias=False)
        self.k_proj   = nn.Linear(d_llm,   d_llm, bias=False)
        self.v_proj   = nn.Linear(d_llm,   d_llm, bias=False)
        self.out_proj = nn.Linear(d_llm,   d_llm, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # patches:    (B*C, N, d_model)
        # prototypes: (n_proto, d_llm)
        BC, N, _ = patches.shape
        P = prototypes.shape[0]

        Q = self.q_proj(patches)                                     # (BC, N, d_llm)
        K = self.k_proj(prototypes).unsqueeze(0).expand(BC, -1, -1) # (BC, P, d_llm)
        V = self.v_proj(prototypes).unsqueeze(0).expand(BC, -1, -1) # (BC, P, d_llm)

        def split_heads(t, seq):
            return t.view(BC, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q, N)   # (BC, H, N, head_dim)
        K = split_heads(K, P)   # (BC, H, P, head_dim)
        V = split_heads(V, P)

        attn = (Q @ K.transpose(-2, -1)) * self.scale               # (BC, H, N, P)
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(BC, N, -1)        # (BC, N, d_llm)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Time-LLM
# ---------------------------------------------------------------------------

class TimeLLM(nn.Module):
    def __init__(self, cfg: TimeLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # RevIN
        self.revin = RevIN(cfg.channels)

        # Patch embedding
        self.patch_embed = PatchEmbedding(cfg.patch_len, cfg.stride, cfg.d_model, cfg.dropout)

        # Frozen GPT-2 backbone
        gpt2 = GPT2Model.from_pretrained(cfg.gpt2_model_name)
        for p in gpt2.parameters():
            p.requires_grad = False
        self.gpt2 = gpt2

        # Word prototypes: evenly-spaced rows from GPT-2 word embedding matrix
        vocab_size = gpt2.wte.weight.size(0)
        idx = torch.linspace(0, vocab_size - 1, steps=cfg.n_text_prototypes).round().long()
        self.register_buffer("word_prototypes", gpt2.wte.weight[idx].detach().clone())

        # Reprogramming layer
        self.reprogramming = ReprogrammingLayer(cfg.d_model, cfg.n_heads, cfg.d_llm, cfg.dropout)

        # Output projection: flatten (num_patches * d_llm) → prediction_length
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.num_patches * cfg.d_llm),
            nn.Linear(cfg.num_patches * cfg.d_llm, cfg.prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # 1. RevIN normalize
        x_norm, mean, std = self.revin.normalize(x)

        # 2. Patch + embed: (B, L, C) → (B*C, num_patches, d_model)
        patches = self.patch_embed(x_norm)

        # 3. Reprogram into LLM space: (B*C, num_patches, d_llm)
        reprogrammed = self.reprogramming(patches, self.word_prototypes)

        # 4. Frozen GPT-2
        gpt_out = self.gpt2(
            inputs_embeds=reprogrammed,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state                                       # (B*C, num_patches, d_llm)

        # 5. Flatten + project: (B*C, num_patches * d_llm) → (B*C, T)
        pred = self.output_proj(gpt_out.flatten(1))              # (B*C, T)

        # 6. Reshape and denormalize: (B, T, C)
        pred = pred.view(B, C, self.cfg.prediction_length).transpose(1, 2)
        return self.revin.denormalize(pred, mean, std)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)
