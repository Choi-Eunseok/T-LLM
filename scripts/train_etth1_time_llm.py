"""
Time-LLM training script for ETTh1 long-term forecasting.

Paper setup (Jin et al., ICLR 2024):
  - Input length L = 96, horizons T ∈ {96, 192, 336, 720}
  - GPT-2 backbone (frozen, no LoRA)
  - patch_len=16, stride=8, d_model=32, n_heads=8
  - Adam lr=1e-4, L1 loss (ETT), checkpoint by val MSE

Note: The original paper uses LLaMA-7B. This script uses GPT-2 for a
fair backbone-controlled comparison with T-LLM (also GPT-2 based).

Paper ETTh1 results (Table 1, LLaMA-7B backbone):
  h=96:  MSE=0.404, MAE=0.422
  h=192: MSE=0.448, MAE=0.448
  h=336: MSE=0.481, MAE=0.466
  h=720: MSE=0.461, MAE=0.462

Usage:
    python scripts/train_etth1_time_llm.py
    python scripts/train_etth1_time_llm.py --horizon 96
    python scripts/train_etth1_time_llm.py --csv data/ETT-small/ETTh1.csv --device cuda
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from time_llm import TimeLLM, TimeLLMConfig
from t_llm.data import load_etth1


# Paper Table 1 reference results (LLaMA-7B backbone)
PAPER_ETTH1 = {
    96:  {"mse": 0.404, "mae": 0.422},
    192: {"mse": 0.448, "mae": 0.448},
    336: {"mse": 0.481, "mae": 0.466},
    720: {"mse": 0.461, "mae": 0.462},
}
PAPER_ETTH1_AVG = {"mse": 0.449, "mae": 0.450}


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


@torch.no_grad()
def evaluate(model: TimeLLM, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    mse_sum = mae_sum = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model.predict(x)
        mse_sum += torch.mean((pred - y) ** 2).item() * x.size(0)
        mae_sum += torch.mean((pred - y).abs()).item() * x.size(0)
        n += x.size(0)
    return mse_sum / max(n, 1), mae_sum / max(n, 1)


# ---------------------------------------------------------------------------
# Single-horizon training run
# ---------------------------------------------------------------------------

def train_horizon(args: argparse.Namespace, horizon: int, device: torch.device) -> dict:
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
    cfg = TimeLLMConfig(
        context_length    = args.context_length,
        prediction_length = horizon,
        channels          = train_set[0][0].shape[-1],
        patch_len         = args.patch_len,
        stride            = args.stride,
        d_model           = args.d_model,
        d_llm             = 768,
        n_heads           = args.n_heads,
        dropout           = 0.1,
        n_text_prototypes = args.n_prototypes,
        gpt2_model_name   = "gpt2",
    )
    model = TimeLLM(cfg).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(
        f"[h={horizon:3d}] params: total={total_params/1e6:.1f}M  "
        f"trainable={trainable_params/1e6:.1f}M",
        flush=True,
    )

    # ---- training loop ----
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
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_n    += x.size(0)

        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"[h={horizon:3d}] epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss/max(train_n,1):.4f}  "
            f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}",
            flush=True,
        )

        if epoch >= args.min_epochs and val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch   = epoch
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        elif epoch >= args.min_epochs:
            no_improve += 1

        if args.patience > 0 and epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_mae = evaluate(model, test_loader, device)
    paper = PAPER_ETTH1.get(horizon, {})
    print(
        f"[h={horizon:3d}] test_mse={test_mse:.4f}  test_mae={test_mae:.4f}  "
        f"(paper/LLaMA-7B: mse={paper.get('mse','?')} mae={paper.get('mae','?')})",
        flush=True,
    )

    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"etth1_time_llm_h{horizon}.pt"
        torch.save({"model": best_state, "cfg": cfg}, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}", flush=True)

    return {
        "horizon":        horizon,
        "best_epoch":     best_epoch,
        "best_val_mse":   best_val_mse,
        "test_mse":       test_mse,
        "test_mae":       test_mae,
        "paper_mse":      paper.get("mse"),
        "paper_mae":      paper.get("mae"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Time-LLM (GPT-2) on ETTh1.")
    p.add_argument("--csv",            type=Path,  default=Path("data/ETT-small/ETTh1.csv"))
    p.add_argument("--horizon",        type=int,   nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--context-length", type=int,   default=96)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--min-epochs",     type=int,   default=3)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--num-workers",    type=int,   default=2)
    # model hyperparameters
    p.add_argument("--patch-len",      type=int,   default=16)
    p.add_argument("--stride",         type=int,   default=8)
    p.add_argument("--d-model",        type=int,   default=32,
                   help="Patch embedding dimension (before reprogramming).")
    p.add_argument("--n-heads",        type=int,   default=8,
                   help="Number of attention heads in the reprogramming layer.")
    p.add_argument("--n-prototypes",   type=int,   default=1000,
                   help="Number of word embedding prototypes used as K/V.")
    p.add_argument("--ckpt-dir",       type=str,   default="checkpoints",
                   help="Directory to save per-horizon checkpoints. Empty string to skip.")
    p.add_argument("--out",            type=Path,  default=Path("results/etth1_time_llm.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"{args.csv} not found. Run train_etth1.py first (it downloads the data)."
        )

    device = get_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  GPU: {torch.cuda.get_device_name(device)}", flush=True)

    if args.ckpt_dir == "":
        args.ckpt_dir = None

    results = [train_horizon(args, h, device) for h in args.horizon]

    avg_mse = sum(r["test_mse"] for r in results) / len(results)
    avg_mae = sum(r["test_mae"] for r in results) / len(results)

    payload = {
        "model":          "Time-LLM (GPT-2 backbone)",
        "dataset":        "ETTh1",
        "context_length": args.context_length,
        "horizons":       args.horizon,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "patch_len":      args.patch_len,
        "stride":         args.stride,
        "d_model":        args.d_model,
        "n_heads":        args.n_heads,
        "n_prototypes":   args.n_prototypes,
        "device":         str(device),
        "results":        results,
        "average": {
            "test_mse":  avg_mse,
            "test_mae":  avg_mae,
            "paper_mse": PAPER_ETTH1_AVG["mse"],
            "paper_mae": PAPER_ETTH1_AVG["mae"],
            "paper_note": "paper values use LLaMA-7B backbone",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {args.out}", flush=True)
    print(f"Average  test_mse={avg_mse:.4f}  test_mae={avg_mae:.4f}", flush=True)
    print(
        f"Paper(LLaMA-7B)  mse={PAPER_ETTH1_AVG['mse']}  mae={PAPER_ETTH1_AVG['mae']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
