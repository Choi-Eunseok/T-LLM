import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TimeLLMConfig:
    context_length: int = 512
    prediction_length: int = 96
    channels: int = 7
    d_model: int = 32
    d_ff: int = 128
    n_heads: int = 8
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    llm_model_name: str = "openai-community/gpt2"
    llm_layers: int = 6
    llm_dim: int = 768
    num_tokens: int = 1000
    description: str = (
        "The Electricity Transformer Temperature (ETT) is a crucial indicator "
        "in the electric power long-term deployment."
    )


class Normalize(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            dims = tuple(range(1, x.ndim - 1))
            self.mean = x.mean(dim=dims, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        if mode == "denorm":
            if self.mean is None or self.stdev is None:
                raise RuntimeError("Normalize must be called with mode='norm' before mode='denorm'.")
            return x * self.stdev + self.mean
        raise ValueError(f"Unknown normalization mode: {mode}")


class ReplicationPad1d(nn.Module):
    def __init__(self, right_padding: int) -> None:
        super().__init__()
        self.right_padding = right_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.right_padding == 0:
            return x
        tail = x[:, :, -1:].repeat(1, 1, self.right_padding)
        return torch.cat([x, tail], dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = ReplicationPad1d(stride)
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return self.dropout(self.value_embedding(x)), n_vars


class FlattenHead(nn.Module):
    def __init__(self, d_ff: int, patch_nums: int, prediction_length: int, dropout: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(d_ff * patch_nums, prediction_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_keys: int, d_llm: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,
        source_embedding: torch.Tensor,
        value_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = target_embedding.shape
        source_length, _ = source_embedding.shape
        heads = self.n_heads

        target = self.query_projection(target_embedding).view(batch, length, heads, -1)
        source = self.key_projection(source_embedding).view(source_length, heads, -1)
        value = self.value_projection(value_embedding).view(source_length, heads, -1)

        scale = 1.0 / math.sqrt(target.shape[-1])
        scores = torch.einsum("blhe,she->bhls", target, source)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attention, value)
        return self.out_projection(out.reshape(batch, length, -1))


class TimeLLM(nn.Module):
    def __init__(self, config: TimeLLMConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from transformers import GPT2Model, GPT2Tokenizer
        except ImportError as exc:
            raise ImportError("TimeLLM requires `pip install transformers`.") from exc

        self.llm_model = GPT2Model.from_pretrained(config.llm_model_name)
        self.llm_model.h = nn.ModuleList(list(self.llm_model.h[: config.llm_layers]))
        self.llm_model.config.n_layer = len(self.llm_model.h)
        self.llm_model.config.output_hidden_states = True
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm_model_name)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        hidden_size = self.llm_model.config.n_embd
        if hidden_size != config.llm_dim:
            raise ValueError(f"GPT-2 hidden size is {hidden_size}, but config.llm_dim={config.llm_dim}.")

        for parameter in self.llm_model.parameters():
            parameter.requires_grad = False

        self.normalize = Normalize(config.channels)
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.dropout)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.mapping_layer = nn.Linear(self.word_embeddings.shape[0], config.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.llm_dim,
            config.dropout,
        )
        self.patch_nums = int((config.context_length - config.patch_len) / config.stride + 2)
        self.output_projection = FlattenHead(
            config.d_ff,
            self.patch_nums,
            config.prediction_length,
            config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x, "norm")
        batch, time, channels = x.shape
        channel_series = x.permute(0, 2, 1).contiguous().reshape(batch * channels, time, 1)

        min_values = channel_series.min(dim=1).values.squeeze(-1)
        max_values = channel_series.max(dim=1).values.squeeze(-1)
        medians = channel_series.median(dim=1).values.squeeze(-1)
        lags = self.calculate_lags(channel_series)
        trends = channel_series.diff(dim=1).sum(dim=1).squeeze(-1)
        prompts = self._build_prompts(min_values, max_values, medians, lags, trends)

        prompt_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids.to(x.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_patched = x.permute(0, 2, 1).contiguous()
        patch_tokens, n_vars = self.patch_embedding(x_patched)
        patch_tokens = self.reprogramming_layer(
            patch_tokens.to(dtype=source_embeddings.dtype),
            source_embeddings,
            source_embeddings,
        )

        prompt_embeddings = prompt_embeddings.to(dtype=patch_tokens.dtype)
        llm_input = torch.cat([prompt_embeddings, patch_tokens], dim=1)
        llm_output = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        llm_output = llm_output[:, :, : self.config.d_ff]
        llm_output = llm_output.reshape(batch, n_vars, llm_output.shape[-2], llm_output.shape[-1])
        llm_output = llm_output.permute(0, 1, 3, 2).contiguous()
        forecast = self.output_projection(llm_output[:, :, :, -self.patch_nums :])
        forecast = forecast.permute(0, 2, 1).contiguous()
        return self.normalize(forecast, "denorm")

    def _build_prompts(
        self,
        min_values: torch.Tensor,
        max_values: torch.Tensor,
        medians: torch.Tensor,
        lags: torch.Tensor,
        trends: torch.Tensor,
    ) -> list[str]:
        prompts = []
        for i in range(min_values.shape[0]):
            trend = "upward" if trends[i].item() > 0 else "downward"
            lag_values = [int(item) for item in lags[i].detach().cpu().tolist()]
            prompts.append(
                "<|start_prompt|>"
                f"Dataset description: {self.config.description} "
                f"Task description: forecast the next {self.config.prediction_length} steps "
                f"given the previous {self.config.context_length} steps information; "
                "Input statistics: "
                f"min value {min_values[i].item()}, "
                f"max value {max_values[i].item()}, "
                f"median value {medians[i].item()}, "
                f"the trend of input is {trend}, "
                f"top 5 lags are : {lag_values}<||>"
            )
        return prompts

    def calculate_lags(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(spectrum * torch.conj(spectrum), dim=-1)
        mean_corr = corr.mean(dim=1)
        _, lags = torch.topk(mean_corr, k=5, dim=-1)
        return lags
