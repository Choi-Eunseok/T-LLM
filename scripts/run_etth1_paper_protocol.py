import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from t_llm import DistillationLoss, TLLM, TLLMConfig
from t_llm.data import make_ett_hour_datasets


PAPER_ETTH1_LONG_TERM_AVERAGE = {"mse": 0.441, "mae": 0.433}

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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--teacher-layers", type=int, default=1)
    parser.add_argument("--student-type", choices=["transformer", "gpt2_lora"], default="transformer")
    parser.add_argument("--student-layers", type=int, default=6)
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/etth1_paper_protocol.json"))
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch tqdm output.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: TLLM, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model.predict(x)
            mse_total += torch.mean((pred - y) ** 2).item() * x.size(0)
            mae_total += torch.mean(torch.abs(pred - y)).item() * x.size(0)
            count += x.size(0)
    return mse_total / max(count, 1), mae_total / max(count, 1)


def run_horizon(args: argparse.Namespace, horizon: int, device: torch.device) -> dict[str, float]:
    seed_everything(args.seed + horizon)
    train_set, valid_set, test_set = make_ett_hour_datasets(
        args.csv,
        context_length=args.context_length,
        prediction_length=horizon,
        stride=1,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

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
    )
    model = TLLM(config).to(device)
    criterion = DistillationLoss(config.d_model).to(device)
    trainable_parameters = [parameter for parameter in list(model.parameters()) + list(criterion.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    best_valid_mse = float("inf")
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
            outputs = model(x, include_teacher=True)
            loss, _ = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)

        valid_mse, valid_mae = evaluate(model, valid_loader, device)
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"horizon={horizon} epoch={epoch} "
            f"train_loss={train_loss / max(train_count, 1):.6f} "
            f"valid_mse={valid_mse:.6f} valid_mae={valid_mae:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, test_loader, device)
    paper_fewshot = PAPER_ETTH1_FEWSHOT_BY_HORIZON[horizon]
    return {
        "horizon": horizon,
        "valid_mse": best_valid_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "paper_fewshot_mse": paper_fewshot["mse"],
        "paper_fewshot_mae": paper_fewshot["mae"],
    }


def main() -> None:
    args = parse_args()
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        "device": str(device),
        "results": results,
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
