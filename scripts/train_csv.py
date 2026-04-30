import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from t_llm import DistillationLoss, DistillationLossConfig, TLLM, TLLMConfig
from t_llm.data import SlidingWindowDataset, make_ett_hour_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train T-LLM on a numeric CSV time series.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--columns", nargs="*", default=None, help="Numeric columns to forecast. Defaults to all numeric columns.")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--teacher-layers", type=int, default=2)
    parser.add_argument("--student-type", choices=["transformer", "gpt2_lora"], default="transformer")
    parser.add_argument("--student-layers", type=int, default=4)
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--student-weight", type=float, default=1.0)
    parser.add_argument("--imitation-weight", type=float, default=1.0)
    parser.add_argument("--guidance-weight", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--split", choices=["random", "ett-hour"], default="random")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/t_llm.pt"))
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch tqdm output.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.split == "ett-hour":
        train_set, valid_set, test_set = make_ett_hour_datasets(
            args.csv,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            target_columns=args.columns,
            stride=args.stride,
        )
    else:
        dataset = SlidingWindowDataset.from_csv(
            args.csv,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            target_columns=args.columns,
            stride=args.stride,
        )
        if len(dataset) < 2:
            raise ValueError("Not enough rows to create train/validation windows.")
        valid_size = max(1, int(len(dataset) * args.valid_ratio))
        train_size = len(dataset) - valid_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        test_set = valid_set

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    sample_x, _ = train_set[0]
    config = TLLMConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
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
    loss_config = DistillationLossConfig(
        teacher_weight=args.teacher_weight,
        student_weight=args.student_weight,
        imitation_weight=args.imitation_weight,
        guidance_weight=args.guidance_weight,
    )
    criterion = DistillationLoss(config.d_model, loss_config).to(device)
    trainable_parameters = [parameter for parameter in list(model.parameters()) + list(criterion.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    best_mse = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=args.quiet)
        train_loss = 0.0
        train_count = 0
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, include_teacher=True)
            loss, parts = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)
            progress.set_postfix({key: f"{value.item():.4f}" for key, value in parts.items()})

        valid_mse, valid_mae = evaluate(model, valid_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss / max(train_count, 1):.6f} "
            f"valid_mse={valid_mse:.6f} valid_mae={valid_mae:.6f}",
            flush=True,
        )
        if valid_mse < best_mse:
            best_mse = valid_mse
            torch.save({"config": config, "model": model.state_dict()}, args.checkpoint)
            print(f"saved {args.checkpoint}", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"test_mse={test_mse:.6f} test_mae={test_mae:.6f}", flush=True)


if __name__ == "__main__":
    main()
