from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DistillationLossConfig:
    teacher_weight: float = 1.0
    student_weight: float = 1.0
    imitation_weight: float = 1.0
    guidance_weight: float = 0.01
    head_guidance_weight: float = 0.5
    tail_guidance_weight: float = 0.5


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistillationLoss(nn.Module):
    def __init__(self, d_model: int, config: DistillationLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or DistillationLossConfig()
        self.student_projection = ProjectionHead(d_model)
        self.teacher_projection = ProjectionHead(d_model)

    def forward(self, outputs: dict[str, object], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_pred = outputs["student_pred"]
        teacher_pred = outputs["teacher_pred"]
        student_features = outputs["student_features"]
        teacher_features = outputs["teacher_features"]

        teacher_loss = F.l1_loss(teacher_pred, target)
        student_loss = F.l1_loss(student_pred, target)
        imitation_loss = F.l1_loss(student_pred, teacher_pred.detach())

        head_loss = F.mse_loss(
            self.student_projection(student_features["head"]),
            self.teacher_projection(teacher_features["head"].detach()),
        )
        tail_loss = F.mse_loss(
            self.student_projection(student_features["tail"]),
            self.teacher_projection(teacher_features["tail"].detach()),
        )
        guidance_loss = (
            self.config.head_guidance_weight * head_loss
            + self.config.tail_guidance_weight * tail_loss
        )

        total = (
            self.config.teacher_weight * teacher_loss
            + self.config.student_weight * student_loss
            + self.config.imitation_weight * imitation_loss
            + self.config.guidance_weight * guidance_loss
        )
        parts = {
            "loss": total.detach(),
            "teacher": teacher_loss.detach(),
            "student": student_loss.detach(),
            "imitation": imitation_loss.detach(),
            "guidance": guidance_loss.detach(),
        }
        return total, parts
