"""
Time-LLM Multi-task 학습 스크립트 — Google Cluster Trace.

Task:
  - ch 0: memory 사용량 예측 (regression, MSE loss)
  - ch 1: 작업 완료 여부 예측 (classification, BCE loss)

Loss = MSE(ch0) + λ * BCE(ch1)

사용법:
    python scripts/train_trace_time_llm.py
    python scripts/train_trace_time_llm.py --csv data/google-cluster/cluster_trace.csv --device cuda
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from time_llm.model import TimeLLM, TimeLLMConfig
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
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
    model: TimeLLM,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> dict:
    model.eval()
    mse_sum = mae_sum = bce_sum = n = 0.0
    tp = fp = tn = fn = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)                         # (B, T, 2)

        mse = nn.functional.mse_loss(pred[:, :, 0], y[:, :, 0]).item()
        mae = nn.functional.l1_loss(pred[:, :, 0],  y[:, :, 0]).item()
        bce = criterion.bce(pred[:, :, 1], y[:, :, 1]).item()
        mse_sum += mse * x.size(0)
        mae_sum += mae * x.size(0)
        bce_sum += bce * x.size(0)
        n       += x.size(0)

        # 분류 정확도: 마지막 time step 기준
        prob     = torch.sigmoid(pred[:, -1, 1])
        label    = y[:, -1, 1].long()
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
        late_ratio        = args.late_ratio,
    )
    loader_kw = dict(
        num_workers        = args.num_workers,
        pin_memory         = (device.type == "cuda"),
        persistent_workers = (args.num_workers > 0),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, **loader_kw)

    # ---- model ----
    cfg = TimeLLMConfig(
        context_length    = args.context_length,
        prediction_length = args.pred_length,
        channels          = 2,              # ch0=memory, ch1=label
        patch_len         = 16,
        stride            = 8,
        d_model           = 32,
        d_llm             = 768,
        n_heads           = 8,
        dropout           = 0.1,
        n_text_prototypes = 1000,
        prompt_token_len  = 32,
        gpt2_model_name   = "gpt2",
        dataset_desc      = (
            "Google Cluster Trace 2019: instance-level CPU/memory usage "
            "from Google data centers sampled every 5 minutes. "
            "Each job is either completed or evicted/killed."
        ),
    )
    model     = TimeLLM(cfg).to(device)
    criterion = MultiTaskLoss(lambda_cls=args.lambda_cls,
                              pos_weight=args.pos_weight)
    criterion.bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in trainable)
    print(f"params: total={total_p/1e6:.1f}M  trainable={trainable_p/1e6:.1f}M")

    # AMP scaler (CUDA only)
    use_amp = (device.type == "cuda") and args.amp
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- training loop ----
    best_val_f1  = -1.0
    best_val_mse = float("inf")
    best_state   = None
    best_epoch   = 0
    no_improve   = 0
    epoch_times  = []
    train_start  = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = train_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)                         # (B, T, 2)
                loss, _ = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)
            train_n    += x.size(0)

        epoch_sec = time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss/max(train_n,1):.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}  "
            f"val_bce={val_metrics['bce']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  val_f1={val_metrics['f1']:.4f}  "
            f"({epoch_sec:.1f}s)",
            flush=True,
        )

        # checkpoint: MSE 기준 (Time-LLM은 전용 분류 head가 없으므로 regression 성능 기준)
        improved = (epoch >= args.min_epochs and
                    val_metrics["mse"] < best_val_mse)
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

    total_train_sec  = time.perf_counter() - train_start
    avg_epoch_sec    = sum(epoch_times) / len(epoch_times)
    gpu_mem_train_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0)

    if best_state is not None:
        model.load_state_dict(best_state)

    # inference time & memory
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, criterion, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_sec           = time.perf_counter() - infer_start
    n_test              = len(test_loader.dataset)
    infer_ms_per_sample = infer_sec / n_test * 1000
    gpu_mem_infer_mb    = (torch.cuda.max_memory_allocated(device) / 1024**2
                           if device.type == "cuda" else 0.0)

    print(
        f"\n[Test]  mse={test_metrics['mse']:.4f}  mae={test_metrics['mae']:.4f}  "
        f"bce={test_metrics['bce']:.4f}  "
        f"acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}",
    )
    print(
        f"  train_total={total_train_sec:.1f}s  "
        f"avg_epoch={avg_epoch_sec:.1f}s  "
        f"infer={infer_ms_per_sample:.3f}ms/sample  "
        f"gpu_train={gpu_mem_train_mb:.0f}MB  gpu_infer={gpu_mem_infer_mb:.0f}MB",
    )

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "trace_time_llm.pt"
        torch.save({"model": best_state, "cfg": cfg}, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

    return {
        "best_epoch":            best_epoch,
        "best_val_f1":           best_val_f1,
        "best_val_mse":          best_val_mse,
        "train_total_sec":       round(total_train_sec, 2),
        "avg_epoch_sec":         round(avg_epoch_sec, 2),
        "infer_ms_per_sample":   round(infer_ms_per_sample, 4),
        "gpu_mem_train_mb":      round(gpu_mem_train_mb, 1),
        "gpu_mem_infer_mb":      round(gpu_mem_infer_mb, 1),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-LLM on Google Cluster Trace.")
    p.add_argument("--csv",            type=Path,  default=Path("data/google-cluster/cluster_trace.csv"))
    p.add_argument("--context-length", type=int,   default=24)
    p.add_argument("--pred-length",    type=int,   default=12)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--lambda-cls",     type=float, default=5.0,
                   help="BCE loss 가중치. MSE(~0.02) 대비 BCE(~0.5) 균형을 위해 기본값 5.0.")
    p.add_argument("--pos-weight",     type=float, default=2.0,
                   help="BCEWithLogitsLoss pos_weight (n_neg/n_pos). 클래스 불균형 보정.")
    p.add_argument("--late-ratio",     type=float, default=0.5,
                   help="job 후반 몇 %% 구간에서만 윈도우 생성 (0.0=전체, 0.5=후반50%%).")
    p.add_argument("--min-epochs",     type=int,   default=3)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         choices=["auto","cpu","cuda","mps"], default="auto")
    p.add_argument("--num-workers",    type=int,   default=2)
    p.add_argument("--amp",            action="store_true",
                   help="Enable AMP (mixed precision, CUDA only).")
    p.add_argument("--split-file",     type=Path,  default=Path("data/google-cluster/split_stratified.json"),
                   help="Stratified instance split 재현용 JSON.")
    p.add_argument("--ckpt-dir",       type=str,   default="checkpoints")
    p.add_argument("--out",            type=Path,  default=Path("results/trace_time_llm.json"))
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
