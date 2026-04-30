import argparse
import json
import math
import random
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from t_llm import DistillationLoss, DistillationLossConfig, TLLM, TLLMConfig
from t_llm.data import make_ett_hour_datasets


PAPER_ETTH1_LONG_TERM_AVERAGE = {"mse": 0.441, "mae": 0.433}
ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

PAPER_ETTH1_FEWSHOT_BY_HORIZON = {
    96: {"mse": 0.396, "mae": 0.412},
    192: {"mse": 0.446, "mae": 0.439},
    336: {"mse": 0.486, "mae": 0.458},
    720: {"mse": 0.498, "mae": 0.479},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETTh1 long-term forecasting protocol from the T-LLM paper.")
    parser.add_argument("--csv", type=Path, default=Path("data/ETT-small/ETTh1.csv"))
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--teacher-layers", type=int, default=2)
    parser.add_argument("--student-type", choices=["transformer", "gpt2_lora"], default="gpt2_lora")
    parser.add_argument("--student-layers", type=int, default=6)
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-dictionary-size", type=int, default=1024)
    parser.add_argument("--no-input-residual", action="store_true", help="Use the paper equations literally without Input Block residuals.")
    parser.add_argument(
        "--spectral-mask-mode",
        choices=["adaptive_stats", "paper_threshold"],
        default="adaptive_stats",
        help="adaptive_stats is more stable with projected spectra; paper_threshold follows Eq. 26 literally.",
    )
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--student-weight", type=float, default=1.0)
    parser.add_argument("--imitation-weight", type=float, default=1.0)
    parser.add_argument("--guidance-weight", type=float, default=0.01)
    parser.add_argument(
        "--selection-branch",
        choices=["teacher_convergence", "teacher", "student", "mean"],
        default="teacher_convergence",
        help=(
            "Which validation signal chooses the checkpoint. teacher_convergence follows the paper's "
            "teacher-convergence idea by ignoring early teacher minima before --min-epochs."
        ),
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=5,
        help="Minimum training epochs before teacher-convergence checkpointing or early stopping can trigger.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stop patience for --selection-branch teacher_convergence. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation MSE improvement counted by teacher-convergence early stopping.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/etth1_paper_protocol.json"))
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch tqdm output.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision to reduce memory use.")
    parser.add_argument("--no-download", action="store_true", help="Fail instead of downloading ETTh1 when --csv is missing.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Use the first fraction of ETTh1 training windows. Set 0.1 for the paper's few-shot protocol.",
    )
    parser.add_argument(
        "--train-sampling",
        choices=["first", "uniform", "random"],
        default="first",
        help="How to select windows when --train-fraction < 1. first matches common chronological few-shot splits.",
    )
    parser.add_argument(
        "--min-selection-updates",
        type=int,
        default=500,
        help=(
            "Minimum optimizer updates before teacher-convergence checkpointing can trigger. "
            "This prevents few-shot runs from stopping after only a few dozen updates."
        ),
    )
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
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot use MPS on this machine.")
    return device


def evaluate(model: TLLM, loader: DataLoader, device: torch.device, branch: str = "student") -> tuple[float, float]:
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if branch == "student":
                pred = model.predict(x)
            elif branch == "teacher":
                outputs = model(x, include_teacher=True)
                pred = outputs["teacher_pred"]
            else:
                raise ValueError(f"Unknown evaluation branch: {branch}")
            mse_total += torch.mean((pred - y) ** 2).item() * x.size(0)
            mae_total += torch.mean(torch.abs(pred - y)).item() * x.size(0)
            count += x.size(0)
    return mse_total / max(count, 1), mae_total / max(count, 1)


def make_grad_scaler(enabled: bool) -> object:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def subset_training_windows(dataset: torch.utils.data.Dataset, fraction: float, mode: str, seed: int) -> torch.utils.data.Dataset:
    if not 0 < fraction <= 1:
        raise ValueError("--train-fraction must be in the range (0, 1].")
    if fraction >= 1.0:
        return dataset

    train_size = max(1, int(len(dataset) * fraction))
    if mode == "first":
        indices = list(range(train_size))
    elif mode == "uniform":
        indices = torch.linspace(0, len(dataset) - 1, steps=train_size).round().long().tolist()
    elif mode == "random":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:train_size].sort().values.tolist()
    else:
        raise ValueError(f"Unknown train sampling mode: {mode}")
    return torch.utils.data.Subset(dataset, indices)


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
    updates_per_epoch = len(train_loader)
    min_update_epochs = 1
    if args.min_selection_updates > 0:
        min_update_epochs = math.ceil(args.min_selection_updates / max(updates_per_epoch, 1))
    effective_min_epochs = min(args.epochs, max(args.min_epochs, min_update_epochs))
    total_planned_updates = updates_per_epoch * args.epochs
    if args.selection_branch == "teacher_convergence" and total_planned_updates < args.min_selection_updates:
        print(
            f"Warning: horizon={horizon} has only {total_planned_updates} planned optimizer updates, "
            f"below --min-selection-updates={args.min_selection_updates}. Increase --epochs for few-shot runs.",
            flush=True,
        )

    sample_x, _ = train_set[0]
    config = TLLMConfig(
        context_length=args.context_length,
        prediction_length=horizon,
        channels=sample_x.shape[-1],
        d_model=args.d_model,
        n_heads=args.heads,
        teacher_layers=args.teacher_layers,
        student_type=args.student_type,
        student_layers=args.student_layers,
        gpt2_model_name=args.gpt2_model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        llm_dictionary_size=args.llm_dictionary_size,
        input_residual=not args.no_input_residual,
        spectral_mask_mode=args.spectral_mask_mode,
    )
    model = TLLM(config).to(device)
    loss_config = DistillationLossConfig(
        teacher_weight=args.teacher_weight,
        student_weight=args.student_weight,
        imitation_weight=args.imitation_weight,
        guidance_weight=args.guidance_weight,
    )
    criterion = DistillationLoss(config.d_model, loss_config).to(device)
    trainable_parameters = [parameter for parameter in list(model.parameters()) + list(criterion.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    best_valid_teacher_mse = float("inf")
    best_epoch = 0
    best_valid_teacher_mae = float("inf")
    best_valid_student_mse = float("inf")
    best_valid_student_mae = float("inf")
    best_selection_mse = float("inf")
    no_improve_epochs = 0
    stopped_epoch = args.epochs
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y in tqdm(
            train_loader,
            desc=f"h={horizon} epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.quiet,
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(x, include_teacher=True)
                loss, _ = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)

        valid_teacher_mse, valid_teacher_mae = evaluate(model, valid_loader, device, branch="teacher")
        valid_student_mse, valid_student_mae = evaluate(model, valid_loader, device, branch="student")
        eligible_for_selection = args.selection_branch != "teacher_convergence" or epoch >= effective_min_epochs
        if args.selection_branch in {"teacher", "teacher_convergence"}:
            selection_mse = valid_teacher_mse
        elif args.selection_branch == "student":
            selection_mse = valid_student_mse
        else:
            selection_mse = 0.5 * (valid_teacher_mse + valid_student_mse)

        improved = selection_mse < best_selection_mse - args.min_delta
        if eligible_for_selection and (best_state is None or improved):
            best_selection_mse = selection_mse
            best_valid_teacher_mse = valid_teacher_mse
            best_epoch = epoch
            best_valid_teacher_mae = valid_teacher_mae
            best_valid_student_mse = valid_student_mse
            best_valid_student_mae = valid_student_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve_epochs = 0
        elif eligible_for_selection:
            no_improve_epochs += 1
        print(
            f"horizon={horizon} epoch={epoch} "
            f"train_loss={train_loss / max(train_count, 1):.6f} "
            f"valid_teacher_mse={valid_teacher_mse:.6f} valid_teacher_mae={valid_teacher_mae:.6f} "
            f"valid_student_mse={valid_student_mse:.6f} valid_student_mae={valid_student_mae:.6f}",
            flush=True,
        )
        if (
            args.selection_branch == "teacher_convergence"
            and args.patience > 0
            and epoch >= effective_min_epochs
            and no_improve_epochs >= args.patience
        ):
            stopped_epoch = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, test_loader, device, branch="student")
    teacher_test_mse, teacher_test_mae = evaluate(model, test_loader, device, branch="teacher")
    paper_fewshot = PAPER_ETTH1_FEWSHOT_BY_HORIZON[horizon]
    result = {
        "horizon": horizon,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_selection_mse": best_selection_mse,
        "best_valid_teacher_mse": best_valid_teacher_mse,
        "best_valid_teacher_mae": best_valid_teacher_mae,
        "best_valid_student_mse": best_valid_student_mse,
        "best_valid_student_mae": best_valid_student_mae,
        "teacher_test_mse": teacher_test_mse,
        "teacher_test_mae": teacher_test_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_windows": len(train_set),
        "optimizer_updates": updates_per_epoch * stopped_epoch,
        "updates_per_epoch": updates_per_epoch,
        "effective_min_epochs": effective_min_epochs,
        "paper_fewshot_mse": paper_fewshot["mse"],
        "paper_fewshot_mae": paper_fewshot["mae"],
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"Using device: {device}", flush=True)

    results = [run_horizon(args, horizon, device) for horizon in args.horizons]
    avg_mse = sum(item["test_mse"] for item in results) / len(results)
    avg_mae = sum(item["test_mae"] for item in results) / len(results)
    payload = {
        "dataset": "ETTh1",
        "context_length": args.context_length,
        "epochs": args.epochs,
        "d_model": args.d_model,
        "student_type": args.student_type,
        "student_layers": args.student_layers,
        "teacher_layers": args.teacher_layers,
        "llm_dictionary_size": args.llm_dictionary_size,
        "input_residual": not args.no_input_residual,
        "spectral_mask_mode": args.spectral_mask_mode,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_fraction": args.train_fraction,
        "train_sampling": args.train_sampling,
        "amp": args.amp,
        "selection_branch": args.selection_branch,
        "min_epochs": args.min_epochs,
        "min_selection_updates": args.min_selection_updates,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "loss_weights": {
            "teacher": args.teacher_weight,
            "student": args.student_weight,
            "imitation": args.imitation_weight,
            "guidance": args.guidance_weight,
        },
        "device": str(device),
        "results": results,
        "paper_references": {
            "average": "Table 1 full-shot ETTh1 average across four horizons.",
            "per_horizon": "Table 3 / Table 9 10% few-shot ETTh1 per-horizon results; compare directly only when train_fraction=0.1.",
        },
        "average": {
            "test_mse": avg_mse,
            "test_mae": avg_mae,
            "paper_long_term_mse": PAPER_ETTH1_LONG_TERM_AVERAGE["mse"],
            "paper_long_term_mae": PAPER_ETTH1_LONG_TERM_AVERAGE["mae"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
