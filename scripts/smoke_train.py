import torch

from t_llm import DistillationLoss, TLLM, TLLMConfig


def main() -> None:
    torch.manual_seed(7)
    config = TLLMConfig(
        context_length=32,
        prediction_length=16,
        channels=4,
        d_model=32,
        n_heads=4,
        teacher_layers=1,
        student_layers=2,
    )
    model = TLLM(config)
    criterion = DistillationLoss(config.d_model)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)

    x = torch.randn(8, config.context_length, config.channels)
    y = torch.randn(8, config.prediction_length, config.channels)
    outputs = model(x)
    loss, parts = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    pred = model.predict(x)
    print(f"loss={loss.item():.4f}")
    print(f"pred_shape={tuple(pred.shape)}")
    print({key: round(value.item(), 4) for key, value in parts.items()})


if __name__ == "__main__":
    main()

