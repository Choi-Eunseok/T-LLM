"""
Google Cluster Trace — T-LLM 예측 결과 시각화.

두 가지 그래프를 생성합니다:
  1. trace_forecast.png : 서로 다른 job instance 8개의 memory 예측 + 분류 결과
  2. trace_cls_result.png : ROC 곡선 + 혼동행렬

각 샘플은 반드시 서로 다른 job(collection_id, instance_index)에서 선택됩니다.
job당 가장 마지막 윈도우(종료 직전 패턴)를 사용합니다.

사용법:
    python scripts/visualize_trace.py \
        --csv  data/google-cluster/cluster_trace.csv \
        --ckpt checkpoints/tllm_trace/trace.pt \
        --device cuda
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader

from t_llm import TLLM, TLLMConfig
from t_llm.data_trace import load_trace

try:
    from time_llm.model import TimeLLM, TimeLLMConfig
    _HAS_TIME_LLM = True
except ImportError:
    _HAS_TIME_LLM = False


# ---------------------------------------------------------------------------
# 모델 타입 공통 인터페이스 래퍼
# ---------------------------------------------------------------------------

class _TimeLLMWrapper:
    """
    TimeLLM을 TLLM과 동일한 인터페이스(predict / predict_cls)로 감싸는 래퍼.
    TimeLLM.forward(x) → (B, T, C): ch0=memory.
    분류는 T-LLM과 동일한 stats 기반 cls_head(predict_cls) 사용.
    """
    def __init__(self, model: "TimeLLM") -> None:
        self._m = model

    def eval(self):
        self._m.eval()
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """메모리 예측 — (B, T, C) 반환 (ch0=memory)."""
        self._m.eval()
        return self._m(x)                   # (B, T, C)

    @torch.no_grad()
    def predict_cls(self, x: torch.Tensor) -> torch.Tensor:
        """완료 로짓 — stats 기반 cls_head (B,)."""
        self._m.eval()
        return self._m.predict_cls(x)       # (B,) raw logit


def _load_model(ckpt_path: Path, device: torch.device):
    """체크포인트에서 cfg 타입을 자동 감지하여 올바른 모델 로드."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    if isinstance(cfg, TLLMConfig):
        model = TLLM(cfg)
        model_label = "T-LLM"
    elif _HAS_TIME_LLM and isinstance(cfg, TimeLLMConfig):
        model = _TimeLLMWrapper(TimeLLM(cfg))
        model_label = "Time-LLM"
    else:
        raise ValueError(f"알 수 없는 config 타입: {type(cfg)}")

    # state_dict 로드 (래퍼인 경우 내부 모델에 적용)
    target = model._m if isinstance(model, _TimeLLMWrapper) else model
    target.load_state_dict(ckpt["model"])
    target.to(device)
    target.eval()

    print(f"모델 로드 [{model_label}]: {ckpt_path}")
    return model, cfg, model_label


# ---------------------------------------------------------------------------
# 예측 수집 (ROC / 혼동행렬용)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model,          # TLLM 또는 _TimeLLMWrapper
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mem_true  : (N, T)  — 실제 memory (StandardScaler 정규화됨)
        mem_pred  : (N, T)  — 예측 memory
        cls_true  : (N,)    — 실제 레이블 (0/1)
        cls_prob  : (N,)    — 예측 완료 확률 (sigmoid)
    """
    model.eval()
    mt, mp, ct, cp = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred_mem  = model.predict(x)                      # (B, T, C)
        cls_logit = model.predict_cls(x)                  # (B,)
        mt.append(y[:, :, 0].cpu().numpy())
        mp.append(pred_mem[:, :, 0].cpu().numpy())
        ct.append(y[:, -1, 1].cpu().numpy().astype(int))
        cp.append(torch.sigmoid(cls_logit).cpu().numpy())
    return (np.concatenate(mt), np.concatenate(mp),
            np.concatenate(ct), np.concatenate(cp))


# ---------------------------------------------------------------------------
# job 단위 샘플 선택
# ---------------------------------------------------------------------------

def pick_samples_by_job(
    test_set,
    model,          # TLLM 또는 _TimeLLMWrapper
    device: torch.device,
    n_completed: int = 4,
    n_evicted:   int = 4,
    seed: int = 0,
) -> list[dict]:
    """
    test_set에서 서로 다른 job(instance)을 골라 샘플을 반환합니다.

    전략:
      1. job_id별로 마지막 윈도우 인덱스를 수집한다
         (마지막 윈도우 = 종료 직전 패턴으로 가장 분류 신호가 강함)
      2. label(1=completed / 0=evicted)로 분리한 뒤 무작위 선택
      3. 모델로 예측 실행
    """
    model.eval()

    # job_id → (마지막 윈도우 인덱스, label) 수집
    job_last: dict[tuple, tuple[int, int]] = {}
    for idx, (jid, (x_arr, y_arr)) in enumerate(
            zip(test_set.job_ids, test_set.samples)):
        label = int(y_arr[-1, 1])
        # 같은 job이면 인덱스를 덮어쓰므로 최종적으로 마지막 윈도우가 남음
        job_last[jid] = (idx, label)

    completed_jobs = [(jid, idx) for jid, (idx, lbl) in job_last.items() if lbl == 1]
    evicted_jobs   = [(jid, idx) for jid, (idx, lbl) in job_last.items() if lbl == 0]

    rng = random.Random(seed)
    rng.shuffle(completed_jobs)
    rng.shuffle(evicted_jobs)

    chosen = (completed_jobs[:n_completed] + evicted_jobs[:n_evicted])

    samples = []
    for jid, idx in chosen:
        x_arr, y_arr = test_set.samples[idx]
        label = int(y_arr[-1, 1])

        # DataLoader 없이 직접 tensor 변환
        x_scaled = test_set._scale(x_arr)
        y_scaled  = test_set._scale(y_arr)

        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mem  = model.predict(x_t)           # (1, T, C)
            cls_logit = model.predict_cls(x_t)       # (1,)
            cls_prob  = torch.sigmoid(cls_logit).item()

        pred_label = int(cls_prob >= 0.5)
        correct    = (pred_label == label)

        samples.append({
            "job_id":     jid,
            "ctx":        x_arr[:, 0],                        # 원시 memory (역정규화용)
            "true":       y_arr[:, 0],
            "pred_mem":   pred_mem[0, :, 0].cpu().numpy(),    # StandardScaler 공간
            "cls_prob":   cls_prob,
            "label":      label,
            "pred_label": pred_label,
            "correct":    correct,
        })

    return samples


# ---------------------------------------------------------------------------
# 그래프 1: 인스턴스별 memory 예측 + 분류 결과
# ---------------------------------------------------------------------------

def plot_forecasts(
    samples:     list[dict],
    scaler,
    out_path:    Path,
    n_cols:      int = 4,
    model_label: str = "T-LLM",
) -> None:
    """
    각 sample subplot:
      (위) 메모리 예측 — context / ground truth / prediction
      (아래) 분류 게이지 바 — 단일 완료 확률 + 실제/예측 레이블

    sample은 반드시 서로 다른 job에서 선택된 것이어야 합니다.
    """
    n_rows = (len(samples) + n_cols - 1) // n_cols
    height_ratios = []
    for _ in range(n_rows):
        height_ratios += [3, 1]       # memory : gauge = 3:1

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5.5))
    fig.suptitle(
        f"{model_label} — Google Cluster Trace  (one window per job instance)",
        fontsize=13, y=1.002,
    )
    gs = gridspec.GridSpec(
        n_rows * 2, n_cols,
        height_ratios=height_ratios,
        hspace=0.65, wspace=0.35,
    )

    L = len(samples[0]["ctx"])
    T = len(samples[0]["true"])
    x_ctx  = np.arange(L)
    x_fore = np.arange(L, L + T)

    for i, s in enumerate(samples):
        r   = i // n_cols
        c   = i  % n_cols
        ax_mem   = fig.add_subplot(gs[r * 2,     c])
        ax_gauge = fig.add_subplot(gs[r * 2 + 1, c])

        # ── 역정규화 ─────────────────────────────────────
        ctx  = scaler.inverse_transform(s["ctx"].reshape(-1, 1)).flatten()
        true = scaler.inverse_transform(s["true"].reshape(-1, 1)).flatten()
        pred = scaler.inverse_transform(s["pred_mem"].reshape(-1, 1)).flatten()

        # ── 메모리 플롯 ──────────────────────────────────
        ax_mem.plot(x_ctx,  ctx,  color="steelblue", lw=1.3, label="context")
        ax_mem.plot(x_fore, true, color="dimgray",   lw=1.2, ls="--",
                    label="ground truth")
        ax_mem.plot(x_fore, pred, color="tomato",    lw=1.6, label="prediction")
        ax_mem.axvline(L, color="black", lw=0.7, ls=":")

        # 실제 결과 마커
        end_x = L + T - 1
        if s["label"] == 1:
            ax_mem.scatter(end_x, true[-1], marker="*", color="green",
                           s=90, zorder=5, label="actual: completed")
        else:
            ax_mem.scatter(end_x, true[-1], marker="X", color="red",
                           s=90, zorder=5, label="actual: evicted")

        # 예측 결과 마커 (▲=completed, ▼=evicted)
        p_marker = "^" if s["pred_label"] == 1 else "v"
        p_color  = "limegreen" if s["pred_label"] == 1 else "salmon"
        ax_mem.scatter(end_x, pred[-1], marker=p_marker, color=p_color,
                       s=70, zorder=5,
                       label=f"pred: {'completed' if s['pred_label']==1 else 'evicted'}")

        actual_str = "Completed" if s["label"] == 1 else "Evicted"
        pred_str   = "Completed" if s["pred_label"] == 1 else "Evicted"
        result_str = "✓ correct" if s["correct"] else "✗ wrong"
        title_color = "green" if s["correct"] else "crimson"
        job_str     = f"job {s['job_id'][0]}-{s['job_id'][1]}"
        ax_mem.set_title(
            f"[{job_str}]\nActual: {actual_str}  |  Pred: {pred_str}  {result_str}",
            fontsize=7, color=title_color,
        )
        ax_mem.set_ylabel("memory", fontsize=7)
        ax_mem.tick_params(labelsize=6)
        if i == 0:
            ax_mem.legend(fontsize=5, loc="upper left", ncol=2)

        # ── 완료 확률 게이지 바 ──────────────────────────
        # x축 0→1, 가로 바 1개 — 실제 레이블에 따라 배경색 설정
        prob = s["cls_prob"]
        bg_color = "honeydew" if s["label"] == 1 else "mistyrose"
        ax_gauge.set_facecolor(bg_color)

        # 확률 채움
        bar_color = "mediumseagreen" if prob >= 0.5 else "tomato"
        ax_gauge.barh([0], [prob], height=0.5, color=bar_color, alpha=0.7)
        ax_gauge.barh([0], [1 - prob], left=[prob], height=0.5,
                      color="lightgray", alpha=0.4)

        # 0.5 threshold 수직선
        ax_gauge.axvline(0.5, color="dimgray", lw=1.0, ls="--")

        # 확률 텍스트
        ax_gauge.text(
            prob + (0.03 if prob < 0.85 else -0.03), 0,
            f"{prob:.2f}",
            va="center", ha="left" if prob < 0.85 else "right",
            fontsize=8, fontweight="bold", color="black",
        )
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.5, 0.5)
        ax_gauge.set_yticks([])
        ax_gauge.set_xlabel("completion probability", fontsize=7)
        ax_gauge.tick_params(labelsize=6)

        # 실제 label 범례 레이블
        actual_bg = "[O] Completed" if s["label"] == 1 else "[X] Evicted"
        ax_gauge.set_title(f"actual outcome: {actual_bg}", fontsize=6.5,
                           color="darkgreen" if s["label"] == 1 else "crimson")

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

    axes[0].plot(fpr, tpr, color="tomato", lw=2,
                 label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

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
    p.add_argument("--ckpt",       type=Path, default=Path("checkpoints/tllm_trace/trace.pt"))
    p.add_argument("--split-file", type=Path, default=Path("data/google-cluster/split_stratified.json"))
    p.add_argument("--context",    type=int,  default=24)
    p.add_argument("--pred-len",   type=int,  default=12)
    p.add_argument("--batch-size", type=int,  default=64)
    p.add_argument("--device",     default="auto")
    p.add_argument("--n-samples",  type=int,  default=8,
                   help="총 시각화 샘플 수 (절반은 completed, 절반은 evicted)")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--out-dir",    type=Path, default=Path("results/plots"))
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 체크포인트 로드 (T-LLM / Time-LLM 자동 감지)
    model, cfg, model_label = _load_model(args.ckpt, device)

    # 데이터 로드 (stratified split 재현)
    _, _, test_set = load_trace(
        args.csv,
        context_length    = args.context,
        prediction_length = args.pred_len,
        split_file        = args.split_file,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    scaler = test_set.scaler

    # ── 전체 예측 수집 (ROC용) ──────────────────────────────
    print("전체 예측 실행 중...")
    mem_true, mem_pred, cls_true, cls_prob = collect_predictions(
        model, test_loader, device
    )
    mse = np.mean((mem_pred - mem_true) ** 2)
    acc = np.mean((cls_prob >= 0.5).astype(int) == cls_true)
    print(f"Memory MSE={mse:.4f}  Classification ACC={acc:.4f}  "
          f"N_test={len(cls_true)}")

    # ── job 단위 다양한 샘플 선택 ───────────────────────────
    half = args.n_samples // 2
    samples = pick_samples_by_job(
        test_set, model, device,
        n_completed = half,
        n_evicted   = half,
        seed        = args.seed,
    )
    print(f"\n선택된 샘플 ({len(samples)}개, job별 1개):")
    for s in samples:
        lbl  = "Completed" if s["label"] == 1 else "Evicted"
        pred = "Completed" if s["pred_label"] == 1 else "Evicted"
        ok   = "✓" if s["correct"] else "✗"
        print(f"  job {s['job_id']}  actual={lbl:9s}  pred={pred:9s}  "
              f"prob={s['cls_prob']:.3f}  {ok}")

    # ── 그래프 생성 ─────────────────────────────────────────
    plot_forecasts(
        samples, scaler,
        out_path    = args.out_dir / f"trace_forecast_{model_label.lower().replace('-','')}.png",
        n_cols      = 4,
        model_label = model_label,
    )
    plot_cls_result(
        cls_true, cls_prob,
        out_path = args.out_dir / f"trace_cls_result_{model_label.lower().replace('-','')}.png",
    )


if __name__ == "__main__":
    main()
