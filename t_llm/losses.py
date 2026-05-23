import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceProjection(nn.Module):
    """Projection head ψ used for intermediate feature guidance (Eq. 24)."""

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
    """
    Total training loss (Eq. 21):

        L = L_teach + λ1 * L_imit + λ2 * L_guide + λ3 * L_stud

    where:
        L_teach = L1(teacher_pred, y)
        L_stud  = L1(student_pred, y)
        L_imit  = L1(student_pred, teacher_pred.detach())
        L_guide = Σ_{k∈K} ω_k · MSE(ψ_S(z_k), ψ_T(e_k).detach())

    Guidance layers K = {early, late} with equal weights ω_k = 0.5.

    Paper hyperparameters for ETT (long-term):
        λ1=1.0, λ2=0.01, λ3=1.0   (Section 4.2)
    """

    def __init__(
        self,
        d_model: int,
        lambda_imit: float  = 1.0,
        lambda_guide: float = 0.01,
        lambda_stud: float  = 1.0,
        noise_std: float    = 0.0,   # > 0 이면 teacher feature에 Gaussian noise 추가
    ) -> None:
        super().__init__()
        self.lambda_imit  = lambda_imit
        self.lambda_guide = lambda_guide
        self.lambda_stud  = lambda_stud
        self.noise_std    = noise_std
        self.student_proj = GuidanceProjection(d_model)
        self.teacher_proj = GuidanceProjection(d_model)

    def forward(
        self,
        outputs: dict[str, object],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s_pred = outputs["student_pred"]
        t_pred = outputs["teacher_pred"]
        s_feat = outputs["student_features"]
        t_feat = outputs["teacher_features"]

        l_teach = F.l1_loss(t_pred, target)
        l_stud  = F.l1_loss(s_pred, target)
        l_imit  = F.l1_loss(s_pred, t_pred.detach())

        # teacher feature noise (학습 중에만 적용)
        if self.noise_std > 0 and self.training:
            t_early = t_feat["early"] + torch.randn_like(t_feat["early"]) * self.noise_std
            t_late  = t_feat["late"]  + torch.randn_like(t_feat["late"])  * self.noise_std
        else:
            t_early = t_feat["early"]
            t_late  = t_feat["late"]

        # guidance at early and late layers (Eq. 24, K = {early, late})
        l_guide = 0.5 * F.mse_loss(
            self.student_proj(s_feat["early"]),
            self.teacher_proj(t_early).detach(),
        ) + 0.5 * F.mse_loss(
            self.student_proj(s_feat["late"]),
            self.teacher_proj(t_late).detach(),
        )

        total = (
            l_teach
            + self.lambda_imit  * l_imit
            + self.lambda_guide * l_guide
            + self.lambda_stud  * l_stud
        )

        parts = {
            "loss":  total.detach(),
            "teach": l_teach.detach(),
            "stud":  l_stud.detach(),
            "imit":  l_imit.detach(),
            "guide": l_guide.detach(),
        }
        return total, parts
