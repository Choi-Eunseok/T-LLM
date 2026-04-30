import torch

from t_llm import DistillationLoss, DistillationLossConfig, TLLM, TLLMConfig


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


def test_input_block_splits_teacher_and_student_tokens() -> None:
    config = TLLMConfig(context_length=16, prediction_length=8, channels=2, d_model=16, n_heads=4)
    model = TLLM(config)
    x = torch.randn(4, config.context_length, config.channels)

    teacher_tokens, student_tokens = model.input_block(x)

    assert teacher_tokens.shape == (4, config.channels, config.d_model)
    assert student_tokens.shape == teacher_tokens.shape
    assert not torch.allclose(teacher_tokens, student_tokens)


def test_spectral_bins_are_selected_from_latent_dimension() -> None:
    config = TLLMConfig(context_length=24, prediction_length=720, d_model=128)

    assert config.spectral_bins == 65


def test_distillation_defaults_match_paper_hyperparameters() -> None:
    config = DistillationLossConfig()

    assert config.student_weight == 1.0
    assert config.imitation_weight == 1.0
    assert config.guidance_weight == 0.01
