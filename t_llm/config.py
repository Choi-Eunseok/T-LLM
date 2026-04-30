from dataclasses import dataclass, field


@dataclass
class TLLMConfig:
    context_length: int = 96
    prediction_length: int = 96
    channels: int = 7
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    teacher_layers: int = 2
    student_type: str = "transformer"
    student_layers: int = 4
    gpt2_model_name: str = "gpt2"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    moving_average_kernel: int = 25
    spectral_capacity_schedule: dict[int, int] = field(
        default_factory=lambda: {24: 16, 48: 24, 96: 32, 192: 48, 336: 64, 720: 96}
    )

    @property
    def spectral_bins(self) -> int:
        eligible = [
            capacity
            for horizon, capacity in sorted(self.spectral_capacity_schedule.items())
            if self.prediction_length <= horizon
        ]
        chosen = eligible[0] if eligible else max(self.spectral_capacity_schedule.values())
        max_bins = self.context_length // 2 + 1
        return max(1, min(chosen, max_bins))
