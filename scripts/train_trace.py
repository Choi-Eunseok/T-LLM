"""
T-LLM Multi-task 학습 스크립트 — Google Cluster Trace.

Task:
  - ch 0: memory 사용량 예측 (regression, MSE loss)
  - ch 1: 작업 완료 여부 예측 (classification, BCE loss)

Loss = MSE(ch0) + λ * BCE(ch1)

ETTh1 비교용 체크포인트는 기존 train_etth1.py로 별도 생성.

사용법:
    python scripts/train_trace.py
    python scripts/train_trace.py --csv data/google-cluster/cluster_trace.csv --device cuda
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from t_llm import TLLM, TLLMConfig
from t_llm.data_trace import load_trace


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


# ---------------------------------------------------------------------------
# Multi-task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    ch 0 → MSE  (memory regression)
    ch 1 → BCE  (completion classification)
    total = mse + lambda_cls * bce

    pos_weight: 클래스 불균형 보정.
      슬라이딩 윈도우 후 중단(0) 작업이 완료(1)보다 많아지는 경향이 있으므로
      pos_weight = n_neg / n_pos 로 설정하면 균형이 맞춰짐.
      기본값 2.0은 중단:완료 ≈ 2:1 비율을 가정.
    """
    def __init__(self, lambda_cls: float = 1.0, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        mse = nn.functional.mse_loss(pred[:, :, 0], target[:, :, 0])
        bce = self.bce(pred[:, :, 1], target[:, :, 1])
        total = mse + self.lambda_cls * bce
        return total, {
            "loss": total.detach(),
            "mse":  mse.detach(),
            "bce":  bce.detach(),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: TLLM,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> dict:
    model.eval()
    mse_sum = mae_sum = bce_sum = n = 0.0
    tp = fp = tn = fn = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model.predict(x)

        mse = nn.functional.mse_loss(pred[:, :, 0], y[:, :, 0]).item()
        mae = nn.functional.l1_loss(pred[:, :, 0],  y[:, :, 0]).item()
        bce = criterion.bce(pred[:, :, 1], y[:, :, 1]).item()
        mse_sum += mse * x.size(0)
        mae_sum += mae * x.size(0)
        bce_sum += bce * x.size(0)
        n       += x.size(0)

        # 분류 정확도: 마지막 time step 기준
        prob  = torch.sigmoid(pred[:, -1, 1])
        label = y[:, -1, 1].long()
        pred_cls = (prob >= 0.5).long()
        tp += ((pred_cls == 1) & (label == 1)).sum().item()
        fp += ((pred_cls == 1) & (label == 0)).sum().item()
        tn += ((pred_cls == 0) & (label == 0)).sum().item()
        fn += ((pred_cls == 0) & (label == 1)).sum().item()

    acc  = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "mse": mse_sum / max(n, 1),
        "mae": mae_sum / max(n, 1),
        "bce": bce_sum / max(n, 1),
        "acc": acc,
        "f1":  f1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, device: torch.device) -> dict:
    seed_everything(args.seed)

    # ---- data ----
    train_set, val_set, test_set = load_trace(
        args.csv,
        context_length    = args.context_length,
        prediction_length = args.pred_length,
        train_ratio       = 0.8,
        val_ratio         = 0.1,
        seed              = args.seed,
        split_file        = args.split_file,
    )
    loader_kw = dict(
        num_workers       = args.num_workers,
        pin_memory        = (device.type == "cuda"),
        persistent_workers= (args.num_workers > 0),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, **loader_kw)

    # ---- model ----
    cfg = TLLMConfig(
        context_length    = args.context_length,
        prediction_length = args.pred_length,
        channels          = 2,          # ch0=memory, ch1=label
        d_model           = 768,
        n_heads           = 4,
        dropout           = 0.1,
        teacher_layers    = 2,
        moving_average_kernel = 25,
        gpt2_model_name   = "gpt2",
        student_layers    = 6,
        lora_rank         = 8,
        lora_alpha        = 16.0,
        lora_dropout      = 0.05,
        dictionary_size   = 1024,
    )
    model     = TLLM(cfg).to(device)
    criterion = MultiTaskLoss(lambda_cls=args.lambda_cls,
                              pos_weight=args.pos_weight)
    criterion.bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )

    # distillation loss (teacher branch)
    from t_llm import DistillationLoss
    distill = DistillationLoss(
        d_model      = cfg.d_model,
        lambda_imit  = 1.0,
        lambda_guide = 0.01,
        lambda_stud  = 1.0,
        noise_std    = args.noise_std,
    ).to(device)

    trainable = [p for p in
                 list(model.parameters()) + list(distill.parameters())
                 if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in trainable)
    print(f"params: total={total_p/1e6:.1f}M  trainable={trainable_p/1e6:.1f}M")

    # ---- training loop ----
    best_val_f1  = -1.0
    best_val_mse = float("inf")
    best_state   = None
    best_epoch   = 0
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(x, teacher=True)

            # Distillation loss on ch0 (memory) ONLY.
            # Applying L1-based L_stud/L_teach to ch1 (binary label) conflicts
            # with BCE and causes the classification head to collapse to the
            # label mean.  Feature guidance (L_guide) is kept as-is since it
            # operates on joint LLM features, not per-channel predictions.
            outputs_mem = {
                "student_pred":     outputs["student_pred"][:, :, 0:1],
                "teacher_pred":     outputs["teacher_pred"][:, :, 0:1],
                "student_features": outputs["student_features"],
                "teacher_features": outputs["teacher_features"],
            }
            dist_loss, _ = distill(outputs_mem, y[:, :, 0:1])

            # Multi-task loss: MSE(ch0) + BCE(ch1)
            mt_loss, _ = criterion(outputs["student_pred"], y)

            loss = dist_loss + mt_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_n    += x.size(0)

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss/max(train_n,1):.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}  "
            f"val_bce={val_metrics['bce']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  val_f1={val_metrics['f1']:.4f}",
            flush=True,
        )

        # checkpoint: F1 기준
        improved = (epoch >= args.min_epochs and
                    val_metrics["f1"] > best_val_f1)
        if improved:
            best_val_f1  = val_metrics["f1"]
            best_val_mse = val_metrics["mse"]
            best_epoch   = epoch
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            no_improve = 0
        elif epoch >= args.min_epochs:
            no_improve += 1

        if args.patience > 0 and epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"\n[Test]  mse={test_metrics['mse']:.4f}  mae={test_metrics['mae']:.4f}  "
        f"bce={test_metrics['bce']:.4f}  "
        f"acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}",
    )

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "trace.pt"
        torch.save({"model": best_state, "cfg": cfg}, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

    return {
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "best_val_mse": best_val_mse,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-LLM on Google Cluster Trace.")
    p.add_argument("--csv",            type=Path,  default=Path("data/google-cluster/cluster_trace.csv"))
    p.add_argument("--context-length", type=int,   default=24)
    p.add_argument("--pred-length",    type=int,   default=12)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--lambda-cls",     type=float, default=5.0,
                   help="BCE loss 가중치. MSE(~0.02) 대비 BCE(~0.5) 균형을 위해 기본값 5.0.")
    p.add_argument("--pos-weight",     type=float, default=2.0,
                   help="BCEWithLogitsLoss pos_weight (n_neg/n_pos). 클래스 불균형 보정.")
    p.add_argument("--min-epochs",     type=int,   default=3)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         choices=["auto","cpu","cuda","mps"], default="auto")
    p.add_argument("--num-workers",    type=int,   default=2)
    p.add_argument("--split-file",     type=Path,  default=Path("data/google-cluster/split.json"),
                   help="Instance split 재현용 JSON.")
    p.add_argument("--ckpt-dir",       type=str,   default="checkpoints")
    p.add_argument("--out",            type=Path,  default=Path("results/trace.json"))
    p.add_argument("--noise-std",      type=float, default=0.0,
                   help="Gaussian noise σ added to teacher features during distillation. 0 = off.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    result = train(args, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
