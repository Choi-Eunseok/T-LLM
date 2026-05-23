from dataclasses import dataclass, field


@dataclass
class TLLMConfig:
    # Data
    context_length: int = 96
    prediction_length: int = 96
    channels: int = 7

    # Model dimensions
    d_model: int = 768  # must match GPT-2 hidden size
    n_heads: int = 4
    dropout: float = 0.1

    # Teacher
    teacher_layers: int = 2
    moving_average_kernel: int = 25
    spectral_capacity_schedule: dict = field(
        default_factory=lambda: {96: 32, 192: 48, 336: 64, 720: 96}
    )

    # Student (GPT-2 + LoRA)
    gpt2_model_name: str = "gpt2"
    student_layers: int = 6
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # Input block
    dictionary_size: int = 1024

    # Classification head (multi-task 용도, 기본 off)
    use_cls_head: bool = False

    @property
    def spectral_bins(self) -> int:
        target = min(
            self.spectral_capacity_schedule,
            key=lambda t: abs(self.prediction_length - t),
        )
        bins = self.spectral_capacity_schedule[target]
        return max(1, min(bins, self.d_model // 2 + 1))
