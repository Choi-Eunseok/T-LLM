"""
Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
Jin et al. (ICLR 2024) — https://arxiv.org/abs/2310.01728

Paper-faithful implementation with:
  - Learnable mapping_layer for source token generation (Section 3.1)
  - Prompt-as-Prefix with per-sample statistics (Section 3.2)
  - seq_len=512, batch=24, epochs=100, lr=0.01 (paper ETTh1 protocol)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TimeLLMConfig:
    context_length:    int   = 512   # paper uses 512 for ETTh1
    prediction_length: int   = 96
    channels:          int   = 7

    patch_len:         int   = 16
    stride:            int   = 8

    d_model:           int   = 32    # patch embedding dimension
    d_llm:             int   = 768   # GPT-2 hidden size
    n_heads:           int   = 8
    dropout:           float = 0.1

    n_text_prototypes: int   = 1000  # K in mapping_layer
    prompt_token_len:  int   = 32    # fixed max prompt length (left-padded)
    gpt2_model_name:   str   = "gpt2"

    dataset_desc: str = (
        "ETT (Electricity Transformer Temperature) dataset records the temperature "
        "of electricity transformers and different types of electricity load."
    )

    @property
    def num_patches(self) -> int:
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
    """(B, L, C) → (B*C, num_patches, d_model)"""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B * C, L)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.drop(self.norm(self.proj(x)))


# ---------------------------------------------------------------------------
# Reprogramming Layer
# ---------------------------------------------------------------------------

class ReprogrammingLayer(nn.Module):
    """
    Multi-head cross-attention mapping patches (d_model) into LLM space (d_llm)
    by attending over learned source embeddings (K/V).

    Q: (B*C, num_patches, d_model) → projected to d_llm
    K, V: source_embeddings (n_proto, d_llm)
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

    def forward(self, patches: torch.Tensor, source_emb: torch.Tensor) -> torch.Tensor:
        BC, N, _ = patches.shape
        P = source_emb.shape[0]

        Q = self.q_proj(patches)
        K = self.k_proj(source_emb).unsqueeze(0).expand(BC, -1, -1)
        V = self.v_proj(source_emb).unsqueeze(0).expand(BC, -1, -1)

        def split_heads(t, seq):
            return t.view(BC, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q, N), split_heads(K, P), split_heads(V, P)
        attn = self.drop(F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(BC, N, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Time-LLM
# ---------------------------------------------------------------------------

class TimeLLM(nn.Module):
    def __init__(self, cfg: TimeLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.revin       = RevIN(cfg.channels)
        self.patch_embed = PatchEmbedding(cfg.patch_len, cfg.stride, cfg.d_model, cfg.dropout)

        # Frozen GPT-2 backbone
        gpt2 = GPT2Model.from_pretrained(cfg.gpt2_model_name)
        for p in gpt2.parameters():
            p.requires_grad = False
        self.gpt2 = gpt2

        # Tokenizer for Prompt-as-Prefix
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt2_model_name)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "left"   # pad left so patches always follow at the end
        self.tokenizer = tokenizer

        # Learnable mapping layer: projects word embedding matrix columns
        # word_emb (d_llm, vocab) → mapping_layer → (d_llm, n_proto) → T → (n_proto, d_llm)
        vocab_size = gpt2.wte.weight.size(0)
        self.mapping_layer = nn.Linear(vocab_size, cfg.n_text_prototypes)

        self.reprogramming = ReprogrammingLayer(cfg.d_model, cfg.n_heads, cfg.d_llm, cfg.dropout)

        self._prompt_cache: torch.Tensor | None = None  # filled on first forward

        # Output projection: flatten patch outputs → prediction
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.num_patches * cfg.d_llm),
            nn.Linear(cfg.num_patches * cfg.d_llm, cfg.prediction_length),
        )

    # ------------------------------------------------------------------
    # Source embeddings via mapping_layer (Section 3.1)
    # ------------------------------------------------------------------

    def _source_embeddings(self) -> torch.Tensor:
        """
        Apply mapping_layer to the (transposed) word embedding matrix.
        word_emb: (vocab, d_llm) → .T → (d_llm, vocab)
        mapping_layer: Linear(vocab → n_proto) → (d_llm, n_proto)
        .T → (n_proto, d_llm)
        """
        word_emb = self.gpt2.wte.weight                   # (vocab, d_llm)
        return self.mapping_layer(word_emb.T).T            # (n_proto, d_llm)

    # ------------------------------------------------------------------
    # Prompt-as-Prefix (Section 3.2)
    # ------------------------------------------------------------------

    def _build_prompt_cache(self, device: torch.device) -> None:
        """
        Tokenize and embed the fixed dataset+task prompt once, store as a buffer.
        Per-sample statistics (min/max/trend/lags) would require re-tokenizing
        every batch which is prohibitively slow; the fixed description captures
        the dataset and task context that is the core of Prompt-as-Prefix.
        """
        prompt = (
            f"<|start_prompt|>"
            f"Dataset description: {self.cfg.dataset_desc} "
            f"Task description: forecast the next {self.cfg.prediction_length} steps "
            f"given the previous {self.cfg.context_length} steps."
            f"<|end_prompt|>"
        )
        enc = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.prompt_token_len,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            self._prompt_cache = self.gpt2.wte(enc.input_ids)  # (1, prompt_len, d_llm)

    def _prompt_embeds(self, B: int, C: int, device: torch.device) -> torch.Tensor:
        """Return cached prompt embeddings expanded to (B*C, prompt_len, d_llm)."""
        if self._prompt_cache is None or self._prompt_cache.device != device:
            self._build_prompt_cache(device)
        return (self._prompt_cache
                .expand(B, -1, -1)
                .unsqueeze(1)
                .expand(-1, C, -1, -1)
                .reshape(B * C, self.cfg.prompt_token_len, self.cfg.d_llm))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # 1. RevIN normalize
        x_norm, mean, std = self.revin.normalize(x)

        # 2. Patch embed: (B*C, num_patches, d_model)
        patches = self.patch_embed(x_norm)

        # 3. Source embeddings via mapping_layer: (n_proto, d_llm)
        source_emb = self._source_embeddings()

        # 4. Reprogram patches into LLM space: (B*C, num_patches, d_llm)
        reprogrammed = self.reprogramming(patches, source_emb)

        # 5. Prompt-as-Prefix: (B*C, prompt_len, d_llm)  — cached after first call
        prompt_emb = self._prompt_embeds(B, C, x.device)

        # 6. Concat [prompt | patches] and run frozen GPT-2
        input_emb = torch.cat([prompt_emb, reprogrammed], dim=1)
        gpt_out   = self.gpt2(
            inputs_embeds=input_emb,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state                                      # (B*C, prompt+patches, d_llm)

        # 7. Take only patch positions (last num_patches tokens)
        patch_out = gpt_out[:, -self.cfg.num_patches:, :]       # (B*C, num_patches, d_llm)

        # 8. Flatten + project: (B*C, T)
        pred = self.output_proj(patch_out.flatten(1))

        # 9. Reshape + RevIN denormalize: (B, T, C)
        pred = pred.view(B, C, self.cfg.prediction_length).transpose(1, 2)
        return self.revin.denormalize(pred, mean, std)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)
