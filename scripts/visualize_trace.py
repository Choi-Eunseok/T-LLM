"""
Google Cluster Trace — T-LLM 예측 결과 시각화.

두 가지 그래프를 생성합니다:
  1. forecast.png : 인스턴스별 memory 사용량 예측 (context + 예측 + 실제)
  2. cls_result.png : 완료 확률 예측 정확도 (ROC 곡선, 혼동행렬)

사용법:
    python scripts/visualize_trace.py \
        --csv  data/google-cluster/cluster_trace.csv \
        --ckpt checkpoints/trace.pt \
        --device cuda
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from t_llm import TLLM, TLLMConfig
from t_llm.data_trace import load_trace


# ---------------------------------------------------------------------------
# 예측 수집
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: TLLM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mem_true  : (N, T)  — 실제 memory (정규화됨)
        mem_pred  : (N, T)  — 예측 memory
        cls_true  : (N,)    — 실제 레이블 (0/1)
        cls_prob  : (N,)    — 예측 완료 확률 (sigmoid)
    """
    model.eval()
    mem_true_list, mem_pred_list, cls_true_list, cls_prob_list = [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model.predict(x)                         # (B, T, 2)

        mem_true_list.append(y[:, :, 0].cpu().numpy())
        mem_pred_list.append(pred[:, :, 0].cpu().numpy())

        # 분류: 마지막 time step의 ch1을 logit으로 사용
        prob = torch.sigmoid(pred[:, -1, 1]).cpu().numpy()
        cls_prob_list.append(prob)
        cls_true_list.append(y[:, -1, 1].cpu().numpy().astype(int))

    return (
        np.concatenate(mem_true_list),
        np.concatenate(mem_pred_list),
        np.concatenate(cls_true_list),
        np.concatenate(cls_prob_list),
    )


# ---------------------------------------------------------------------------
# 그래프 1: 인스턴스별 memory 예측
# ---------------------------------------------------------------------------

def plot_forecasts(
    test_loader: DataLoader,
    model: TLLM,
    device: torch.device,
    scaler,
    out_path: Path,
    n_samples: int = 8,
) -> None:
    """
    서로 다른 job에서 샘플을 뽑아 시각화.
    각 샘플마다 두 개 subplot:
      (위) 메모리 예측 — context / ground truth / prediction
      (아래) 예측 완료 확률 시계열 — 0.5 threshold 표시
    제목에 실제 결과 vs 예측 결과(정답/오답) 표시.
    """
    model.eval()
    completed_samples, failed_samples = [], []
    seen_fps: set = set()          # job fingerprint — 같은 job 중복 방지

    with torch.no_grad():
        for x, y in test_loader:
            if (len(completed_samples) >= n_samples // 2 and
                    len(failed_samples) >= n_samples // 2):
                break
            x, y = x.to(device), y.to(device)
            pred = model.predict(x)

            for b in range(x.size(0)):
                ctx_np = x[b, :, 0].cpu().numpy()
                # job fingerprint: 처음 5 스텝 값 (반올림)
                fp = tuple(np.round(ctx_np[:5], 5))
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)

                label       = int(y[b, -1, 1].item())
                prob_series = torch.sigmoid(pred[b, :, 1]).cpu().numpy()  # (T,)
                pred_label  = int(prob_series[-1] >= 0.5)
                correct     = (pred_label == label)

                sample = {
                    "ctx":        ctx_np,
                    "true":       y[b, :, 0].cpu().numpy(),
                    "pred_mem":   pred[b, :, 0].cpu().numpy(),
                    "prob_series": prob_series,
                    "label":      label,
                    "pred_label": pred_label,
                    "correct":    correct,
                }
                if label == 1 and len(completed_samples) < n_samples // 2:
                    completed_samples.append(sample)
                elif label == 0 and len(failed_samples) < n_samples // 2:
                    failed_samples.append(sample)

    samples = completed_samples + failed_samples
    if not samples:
        print("No samples to visualize")
        return

    n_cols      = 4
    n_sample_rows = (len(samples) + n_cols - 1) // n_cols
    # 각 sample-row: 메모리(height 2) + 확률(height 1)
    height_ratios = []
    for _ in range(n_sample_rows):
        height_ratios += [2, 1]

    fig = plt.figure(figsize=(n_cols * 5, n_sample_rows * 4.5 + 0.5))
    fig.suptitle("T-LLM — Google Cluster Trace Forecast", fontsize=13, y=1.005)
    gs  = matplotlib.gridspec.GridSpec(
        n_sample_rows * 2, n_cols,
        height_ratios=height_ratios,
        hspace=0.55, wspace=0.35,
    )

    L = len(samples[0]["ctx"])
    T = len(samples[0]["true"])
    x_ctx  = np.arange(L)
    x_fore = np.arange(L, L + T)

    for i, s in enumerate(samples):
        row_grp = i // n_cols
        col     = i  % n_cols
        ax_mem  = fig.add_subplot(gs[row_grp * 2,     col])
        ax_prob = fig.add_subplot(gs[row_grp * 2 + 1, col])

        # ── 메모리 역정규화 ──────────────────────────────
        ctx  = scaler.inverse_transform(s["ctx"].reshape(-1, 1)).flatten()
        true = scaler.inverse_transform(s["true"].reshape(-1, 1)).flatten()
        pred = scaler.inverse_transform(s["pred_mem"].reshape(-1, 1)).flatten()

        ax_mem.plot(x_ctx,  ctx,  color="steelblue", lw=1.2, label="context")
        ax_mem.plot(x_fore, true, color="gray",      lw=1.2, ls="--", label="ground truth")
        ax_mem.plot(x_fore, pred, color="tomato",    lw=1.5, label="prediction")
        ax_mem.axvline(L, color="black", lw=0.7, ls=":")

        # 실제 결과 마커 (예측 구간 마지막 점)
        end_x = L + T - 1
        if s["label"] == 1:
            ax_mem.scatter(end_x, true[-1], marker="*", color="green",
                           s=80, zorder=5, label="actual: completed")
        else:
            ax_mem.scatter(end_x, true[-1], marker="X", color="red",
                           s=80, zorder=5, label="actual: evicted")

        # 예측 결과 마커
        pred_marker = "^" if s["pred_label"] == 1 else "v"
        pred_color  = "limegreen" if s["pred_label"] == 1 else "salmon"
        ax_mem.scatter(end_x, pred[-1], marker=pred_marker,
                       color=pred_color, s=60, zorder=5,
                       label=f"pred: {'completed' if s['pred_label']==1 else 'evicted'}")

        actual_str = "Completed" if s["label"] == 1 else "Evicted"
        pred_str   = "Completed" if s["pred_label"] == 1 else "Evicted"
        result_str = "✓ correct" if s["correct"] else "✗ wrong"
        title_color = "green" if s["correct"] else "crimson"
        ax_mem.set_title(
            f"Actual: {actual_str}  |  Pred: {pred_str}  {result_str}",
            fontsize=7.5, color=title_color,
        )
        ax_mem.set_ylabel("memory", fontsize=7)
        ax_mem.tick_params(labelsize=6)
        if i == 0:
            ax_mem.legend(fontsize=5.5, loc="upper left")

        # ── 완료 확률 시계열 ─────────────────────────────
        prob = s["prob_series"]
        ax_prob.plot(x_fore, prob, color="darkorange", lw=1.5)
        ax_prob.fill_between(x_fore, prob, 0.5,
                             where=(prob >= 0.5), alpha=0.25, color="green",
                             label="pred: completed")
        ax_prob.fill_between(x_fore, prob, 0.5,
                             where=(prob <  0.5), alpha=0.25, color="red",
                             label="pred: evicted")
        ax_prob.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax_prob.set_ylim(0, 1)
        ax_prob.set_xlabel("time step (5 min)", fontsize=7)
        ax_prob.set_ylabel("comp. prob", fontsize=7)
        ax_prob.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path}")


# ---------------------------------------------------------------------------
# 그래프 2: 분류 성능 (ROC + 혼동행렬)
# ---------------------------------------------------------------------------

def plot_cls_result(
    cls_true: np.ndarray,
    cls_prob: np.ndarray,
    out_path: Path,
) -> None:
    from sklearn.metrics import (
        roc_curve, auc,
        confusion_matrix, ConfusionMatrixDisplay,
        classification_report,
    )

    pred_label = (cls_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(cls_true, cls_prob)
    roc_auc = auc(fpr, tpr)
    acc = (pred_label == cls_true).mean()
    cm  = confusion_matrix(cls_true, pred_label)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"T-LLM Job Completion Prediction  |  ACC={acc:.3f}  AUC={roc_auc:.3f}",
        fontsize=12,
    )

    # ROC 곡선
    axes[0].plot(fpr, tpr, color="tomato", lw=2,
                 label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # 혼동행렬
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Evicted (0)", "Completed (1)"],
    )
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path}")

    print("\n[Classification Report]")
    print(classification_report(cls_true, pred_label,
                                target_names=["Evicted(0)", "Completed(1)"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        type=Path, default=Path("data/google-cluster/cluster_trace.csv"))
    p.add_argument("--ckpt",       type=Path, default=Path("checkpoints/trace.pt"))
    p.add_argument("--split-file", type=Path, default=Path("data/google-cluster/split.json"))
    p.add_argument("--context",    type=int,  default=24)
    p.add_argument("--pred-len",   type=int,  default=12)
    p.add_argument("--batch-size", type=int,  default=64)
    p.add_argument("--device",     default="auto")
    p.add_argument("--n-samples",  type=int,  default=8)
    p.add_argument("--out-dir",    type=Path, default=Path("results/plots"))
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: TLLMConfig = ckpt["cfg"]
    model = TLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"모델 로드: {args.ckpt}")

    # 데이터 로드 (split.json으로 동일한 test set 재현)
    _, _, test_set = load_trace(
        args.csv,
        context_length    = args.context,
        prediction_length = args.pred_len,
        split_file        = args.split_file,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    scaler = test_set.scaler

    # 예측 수집
    print("예측 실행 중...")
    mem_true, mem_pred, cls_true, cls_prob = collect_predictions(model, test_loader, device)

    mse = np.mean((mem_pred - mem_true) ** 2)
    acc = np.mean((cls_prob >= 0.5).astype(int) == cls_true)
    print(f"Memory MSE={mse:.4f}  Classification ACC={acc:.4f}")

    # 그래프 생성
    plot_forecasts(
        test_loader, model, device, scaler,
        out_path  = args.out_dir / "trace_forecast.png",
        n_samples = args.n_samples,
    )
    plot_cls_result(
        cls_true, cls_prob,
        out_path = args.out_dir / "trace_cls_result.png",
    )


if __name__ == "__main__":
    main()
