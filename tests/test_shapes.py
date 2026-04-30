import torch

from t_llm import DistillationLoss, TLLM, TLLMConfig


def test_tllm_forward_and_loss_shapes() -> None:
    config = TLLMConfig(
        context_length=24,
        prediction_length=12,
        channels=3,
        d_model=16,
        n_heads=4,
        teacher_layers=1,
        student_layers=2,
    )
    model = TLLM(config)
    criterion = DistillationLoss(config.d_model)

    x = torch.randn(2, config.context_length, config.channels)
    y = torch.randn(2, config.prediction_length, config.channels)
    outputs = model(x)
    loss, parts = criterion(outputs, y)

    assert outputs["student_pred"].shape == y.shape
    assert outputs["teacher_pred"].shape == y.shape
    assert loss.ndim == 0
    assert set(parts) == {"loss", "teacher", "student", "imitation", "guidance"}


def test_predict_uses_student_only() -> None:
    config = TLLMConfig(context_length=16, prediction_length=8, channels=2, d_model=16, n_heads=4)
    model = TLLM(config)
    x = torch.randn(4, config.context_length, config.channels)

    pred = model.predict(x)

    assert pred.shape == (4, config.prediction_length, config.channels)

