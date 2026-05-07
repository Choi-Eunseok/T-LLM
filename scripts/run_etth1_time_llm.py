import argparse
import json
import random
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from t_llm.data import make_ett_hour_datasets
from t_llm.time_llm import TimeLLM, TimeLLMConfig


ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
TIME_LLM_ETTH1_LR = {96: 0.01, 192: 0.02, 336: 0.001, 720: 0.01}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GPT-2 Time-LLM baseline on ETTh1.")
    parser.add_argument("--csv", type=Path, default=Path("data/ETT-small/ETTh1.csv"))
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=None, help="Override horizon-specific Time-LLM ETTh1 LR.")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--llm-model-name", default="openai-community/gpt2")
    parser.add_argument("--llm-layers", type=int, default=6)
    parser.add_argument("--llm-dim", type=int, default=768)
    parser.add_argument("--num-tokens", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-sampling", choices=["first", "uniform", "random"], default="first")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out", type=Path, default=Path("results/etth1_time_llm.json"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_etth1(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(ETTH1_URL) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_path.replace(path)


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    return device


def subset_training_windows(dataset: torch.utils.data.Dataset, fraction: float, mode: str, seed: int) -> torch.utils.data.Dataset:
    if not 0 < fraction <= 1:
        raise ValueError("--train-fraction must be in the range (0, 1].")
    if fraction >= 1:
        return dataset
    train_size = max(1, int(len(dataset) * fraction))
    if mode == "first":
        indices = list(range(train_size))
    elif mode == "uniform":
        indices = torch.linspace(0, len(dataset) - 1, steps=train_size).round().long().tolist()
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:train_size].sort().values.tolist()
    return torch.utils.data.Subset(dataset, indices)


def autocast_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def evaluate(
    model: TimeLLM,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                pred = model(x)
            mse_total += torch.mean((pred - y) ** 2).item() * x.size(0)
            mae_total += torch.mean(torch.abs(pred - y)).item() * x.size(0)
            count += x.size(0)
    model.train()
    return mse_total / max(count, 1), mae_total / max(count, 1)


def make_grad_scaler(enabled: bool) -> object:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_horizon(args: argparse.Namespace, horizon: int, device: torch.device) -> dict[str, float]:
    seed_everything(args.seed + horizon)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_set, valid_set, test_set = make_ett_hour_datasets(
        args.csv,
        context_length=args.context_length,
        prediction_length=horizon,
        stride=1,
    )
    train_set = subset_training_windows(train_set, args.train_fraction, args.train_sampling, args.seed + horizon)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_kwargs)

    sample_x, _ = train_set[0]
    config = TimeLLMConfig(
        context_length=args.context_length,
        prediction_length=horizon,
        channels=sample_x.shape[-1],
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.heads,
        dropout=args.dropout,
        patch_len=args.patch_len,
        stride=args.stride,
        llm_model_name=args.llm_model_name,
        llm_layers=args.llm_layers,
        llm_dim=args.llm_dim,
        num_tokens=args.num_tokens,
    )
    model = TimeLLM(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = args.lr if args.lr is not None else TIME_LLM_ETTH1_LR[horizon]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = autocast_dtype(args.amp_dtype)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    criterion = torch.nn.MSELoss()

    best_valid_mse = float("inf")
    best_valid_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    stopped_epoch = args.epochs
    updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y in tqdm(train_loader, desc=f"TimeLLM h={horizon} epoch {epoch}/{args.epochs}", leave=False, disable=args.quiet):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
                loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at horizon={horizon}, epoch={epoch}. "
                    "Try disabling --amp or lowering --lr."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)
            updates += 1

        valid_mse, valid_mae = evaluate(model, valid_loader, device, use_amp, amp_dtype)
        print(
            f"horizon={horizon} epoch={epoch} train_mse={train_loss / max(train_count, 1):.6f} "
            f"valid_mse={valid_mse:.6f} valid_mae={valid_mae:.6f}",
            flush=True,
        )
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_valid_mae = valid_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
            stopped_epoch = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, test_loader, device, use_amp, amp_dtype)
    result = {
        "horizon": horizon,
        "lr": lr,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_valid_mse": best_valid_mse,
        "best_valid_mae": best_valid_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_windows": len(train_set),
        "optimizer_updates": updates,
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["peak_cuda_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
    return result


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        if args.no_download:
            raise FileNotFoundError(f"{args.csv} does not exist.")
        print(f"Downloading ETTh1 to {args.csv}", flush=True)
        download_etth1(args.csv)

    device = select_device(args.device)
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"Using device: {device}", flush=True)

    results = [run_horizon(args, horizon, device) for horizon in args.horizons]
    payload = {
        "model": "Time-LLM",
        "dataset": "ETTh1",
        "context_length": args.context_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_fraction": args.train_fraction,
        "train_sampling": args.train_sampling,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "llm_model_name": args.llm_model_name,
        "llm_layers": args.llm_layers,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype if args.amp else None,
        "device": str(device),
        "results": results,
        "average": {
            "test_mse": sum(item["test_mse"] for item in results) / len(results),
            "test_mae": sum(item["test_mae"] for item in results) / len(results),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
