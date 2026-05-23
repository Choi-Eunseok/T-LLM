"""
T-LLM training script for ETTh1 long-term forecasting.

Paper setup (Section 4.2 / 4.3):
  - Input length L = 96, horizons T ∈ {96, 192, 336, 720}
  - GPT-2 backbone (first 6 layers) + LoRA
  - Adam lr = 5e-4, L1 loss (ETT)
  - λ1=1.0 (imitation), λ2=0.01 (guidance), λ3=1.0 (student)
  - Early stopping / checkpoint by teacher validation MSE convergence

Usage:
    python scripts/train_etth1.py
    python scripts/train_etth1.py --horizon 96
    python scripts/train_etth1.py --csv data/ETT-small/ETTh1.csv --device cuda
"""

import argparse
import json
import random
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from t_llm import TLLM, TLLMConfig, DistillationLoss
from t_llm.data import load_etth1


# Paper Table 1 reference results (average over 4 horizons)
PAPER_ETTH1_AVG = {"mse": 0.441, "mae": 0.433}
# Paper Table 1 per-horizon results
PAPER_ETTH1 = {
    96:  {"mse": 0.417, "mae": 0.419},
    192: {"mse": 0.439, "mae": 0.432},
    336: {"mse": 0.458, "mae": 0.445},
    720: {"mse": 0.449, "mae": 0.434},
}

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


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


def download_etth1(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(ETTH1_URL) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(path)


@torch.no_grad()
def evaluate(model: TLLM, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return student MSE and MAE on a DataLoader."""
    model.eval()
    mse_sum = mae_sum = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model.predict(x)
        mse_sum += torch.mean((pred - y) ** 2).item() * x.size(0)
        mae_sum += torch.mean((pred - y).abs()).item() * x.size(0)
        n += x.size(0)
    return mse_sum / max(n, 1), mae_sum / max(n, 1)


@torch.no_grad()
def evaluate_teacher(model: TLLM, loader: DataLoader, device: torch.device) -> float:
    """Return teacher validation MSE (used for checkpoint selection)."""
    mse_sum = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t_pred = model.predict_teacher(x)
        mse_sum += torch.mean((t_pred - y) ** 2).item() * x.size(0)
        n += x.size(0)
    return mse_sum / max(n, 1)


# ---------------------------------------------------------------------------
# Single-horizon training run
# ---------------------------------------------------------------------------

def train_horizon(
    args: argparse.Namespace,
    horizon: int,
    device: torch.device,
) -> dict:
    seed_everything(args.seed + horizon)

    # ---- data ----
    train_set, val_set, test_set = load_etth1(args.csv, args.context_length, horizon)
    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, **loader_kw)

    # ---- model ----
    cfg = TLLMConfig(
        context_length    = args.context_length,
        prediction_length = horizon,
        channels          = train_set[0][0].shape[-1],
        d_model           = 768,   # GPT-2 hidden size
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
    criterion = DistillationLoss(
        d_model      = cfg.d_model,
        lambda_imit  = 1.0,
        lambda_guide = 0.01,
        lambda_stud  = 1.0,
        noise_std    = args.noise_std,
    ).to(device)

    trainable = [p for p in list(model.parameters()) + list(criterion.parameters())
                 if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # ---- training loop ----
    best_teacher_mse = float("inf")
    best_student_mse = float("inf")
    best_state       = None
    best_epoch       = 0
    no_improve       = 0
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
            outputs = model(x, teacher=True)
            loss, parts = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_n    += x.size(0)

        epoch_sec = time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)

        # validation
        teacher_val_mse = evaluate_teacher(model, val_loader, device)
        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"[h={horizon:3d}] epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss/max(train_n,1):.4f}  "
            f"t_val={teacher_val_mse:.4f}  "
            f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  "
            f"({epoch_sec:.1f}s)",
            flush=True,
        )

        # Checkpoint selection: best teacher val MSE (paper Section 3.6).
        # "Early stopping determined by convergence of the temporal teacher."
        if epoch >= args.min_epochs and teacher_val_mse < best_teacher_mse:
            best_teacher_mse = teacher_val_mse
            best_student_mse = val_mse
            best_epoch       = epoch
            best_state       = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve       = 0
        elif epoch >= args.min_epochs:
            no_improve += 1

        if args.patience > 0 and epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    total_train_sec    = time.perf_counter() - train_start
    avg_epoch_sec      = sum(epoch_times) / len(epoch_times)
    gpu_mem_train_mb   = (torch.cuda.max_memory_allocated(device) / 1024**2
                          if device.type == "cuda" else 0.0)

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # inference time & memory
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.perf_counter()
    test_mse, test_mae = evaluate(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_sec          = time.perf_counter() - infer_start
    n_test             = len(test_loader.dataset)
    infer_ms_per_sample = infer_sec / n_test * 1000
    gpu_mem_infer_mb   = (torch.cuda.max_memory_allocated(device) / 1024**2
                          if device.type == "cuda" else 0.0)

    paper = PAPER_ETTH1.get(horizon, {})
    print(
        f"[h={horizon:3d}] test_mse={test_mse:.4f}  test_mae={test_mae:.4f}  "
        f"(paper: mse={paper.get('mse','?')} mae={paper.get('mae','?')})",
        flush=True,
    )
    print(
        f"  train_total={total_train_sec:.1f}s  "
        f"avg_epoch={avg_epoch_sec:.1f}s  "
        f"infer={infer_ms_per_sample:.3f}ms/sample  "
        f"gpu_train={gpu_mem_train_mb:.0f}MB  gpu_infer={gpu_mem_infer_mb:.0f}MB",
        flush=True,
    )

    # optionally save checkpoint
    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"etth1_h{horizon}.pt"
        torch.save({"model": best_state, "cfg": cfg}, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}", flush=True)

    return {
        "horizon":               horizon,
        "best_epoch":            best_epoch,
        "best_teacher_mse":      best_teacher_mse,
        "best_student_mse":      best_student_mse,
        "test_mse":              test_mse,
        "test_mae":              test_mae,
        "paper_mse":             paper.get("mse"),
        "paper_mae":             paper.get("mae"),
        "train_total_sec":       round(total_train_sec, 2),
        "avg_epoch_sec":         round(avg_epoch_sec, 2),
        "infer_ms_per_sample":   round(infer_ms_per_sample, 4),
        "gpu_mem_train_mb":      round(gpu_mem_train_mb, 1),
        "gpu_mem_infer_mb":      round(gpu_mem_infer_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train T-LLM on ETTh1.")
    p.add_argument("--csv",            type=Path,   default=Path("data/ETT-small/ETTh1.csv"))
    p.add_argument("--horizon",        type=int,    nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--context-length", type=int,    default=96)
    p.add_argument("--epochs",         type=int,    default=50)
    p.add_argument("--batch-size",     type=int,    default=64)
    p.add_argument("--lr",             type=float,  default=5e-4)
    p.add_argument("--min-epochs",     type=int,    default=10,
                   help="Minimum epochs before early stopping / checkpoint selection.")
    p.add_argument("--patience",       type=int,    default=10,
                   help="Early-stop patience on teacher val MSE (0 = disabled).")
    p.add_argument("--seed",           type=int,    default=42)
    p.add_argument("--device",         choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--num-workers",    type=int,    default=2)
    p.add_argument("--no-download",    action="store_true",
                   help="Raise an error instead of downloading ETTh1 if --csv is missing.")
    p.add_argument("--ckpt-dir",       type=str,    default="checkpoints",
                   help="Directory to save per-horizon checkpoints. Set empty string to skip.")
    p.add_argument("--out",            type=Path,   default=Path("results/etth1.json"),
                   help="Path to write JSON results.")
    p.add_argument("--noise-std",      type=float,  default=0.0,
                   help="Gaussian noise σ added to teacher features during distillation. 0 = off.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        if args.no_download:
            raise FileNotFoundError(args.csv)
        print(f"Downloading ETTh1 to {args.csv} …", flush=True)
        download_etth1(args.csv)

    device = get_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  GPU: {torch.cuda.get_device_name(device)}", flush=True)

    # convert empty string to None for ckpt_dir
    if args.ckpt_dir == "":
        args.ckpt_dir = None

    results = [train_horizon(args, h, device) for h in args.horizon]

    avg_mse = sum(r["test_mse"] for r in results) / len(results)
    avg_mae = sum(r["test_mae"] for r in results) / len(results)
    payload = {
        "dataset":       "ETTh1",
        "context_length": args.context_length,
        "horizons":      args.horizon,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch_size":    args.batch_size,
        "device":        str(device),
        "results":       results,
        "average": {
            "test_mse":        avg_mse,
            "test_mae":        avg_mae,
            "paper_mse":       PAPER_ETTH1_AVG["mse"],
            "paper_mae":       PAPER_ETTH1_AVG["mae"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {args.out}", flush=True)
    print(f"Average  test_mse={avg_mse:.4f}  test_mae={avg_mae:.4f}", flush=True)
    print(f"Paper    test_mse={PAPER_ETTH1_AVG['mse']}  test_mae={PAPER_ETTH1_AVG['mae']}", flush=True)


if __name__ == "__main__":
    main()
