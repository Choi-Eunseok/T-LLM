# T-LLM vs Time-LLM — Forecasting & Job Completion Prediction

PyTorch reimplementation of two LLM-based time series forecasting methods, evaluated on:
- **ETTh1** long-term forecasting (4 horizons)
- **Google Cluster Trace 2019** memory forecasting + job completion classification (multi-task)

---

## Results

### ETTh1 Long-term Forecasting (MSE ↓, MAE ↓)

| Model | Backbone | h=96 | h=192 | h=336 | h=720 | **Avg MSE** |
|-------|----------|------|-------|-------|-------|-------------|
| **T-LLM** (ours) | GPT-2 + LoRA | 0.3937 | **0.4440** | **0.4908** | 0.5308 | 0.4648 |
| **T-LLM + Noisy** (ours) | GPT-2 + LoRA | **0.3875** | 0.4711 | 0.5024 | 0.5171 | 0.4695 |
| **Time-LLM** (ours) | GPT-2 frozen | 0.3908 | 0.4461 | 0.4992 | **0.5018** | **0.4595** |
| T-LLM (paper) | GPT-2 + LoRA | 0.417 | 0.439 | 0.458 | 0.449 | 0.441 |
| Time-LLM (paper) | LLaMA-7B | 0.404 | 0.448 | 0.481 | 0.461 | 0.449 |

> Paper results for Time-LLM use LLaMA-7B. Our Time-LLM run uses GPT-2 for a fair backbone-controlled comparison.

### Google Cluster Trace — Memory Forecasting + Job Completion (Multi-task)

| Model | Memory MSE ↓ | Memory MAE ↓ | Completion ACC ↑ | Completion F1 ↑ |
|-------|-------------|-------------|-----------------|----------------|
| T-LLM (ours) | 0.1707 | **0.0884** | **0.712** | **0.752** |
| T-LLM + Noisy (ours) | 0.1718 | 0.0902 | 0.710 | 0.751 |
| **Time-LLM** (ours) | **0.1630** | 0.0992 | 0.695 | 0.742 |

> All three models use the **same stats-based classification head** (337 params, computed from pre-RevIN memory statistics). The label channel (constant 0/1 per job) is destroyed by RevIN, so completion is predicted from raw memory level/trend statistics rather than the forecasting backbone. Classification scores are therefore similar across models; the meaningful comparison is **memory forecasting**, where Time-LLM wins on MSE and T-LLM on MAE.

### Training Efficiency (ETTh1, all 4 horizons combined)

| Model | Total Train Time | Per-epoch | Inference | GPU (train) | GPU (infer) |
|-------|-----------------|-----------|-----------|-------------|-------------|
| **T-LLM** | **~10.5 min** | **6–7 s** | **0.23 ms/sample** | **2.5 GB** | **1.3 GB** |
| **T-LLM + Noisy** | ~10.8 min | 6–7 s | 0.22 ms/sample | 2.5 GB | 1.3 GB |
| Time-LLM | ~40.5 h | 950–1190 s | 17.9 ms/sample | 20.2 GB | 2.9 GB |

T-LLM trains **231× faster** and runs inference **78× faster** than Time-LLM, using **8× less GPU memory**.

### Training Efficiency (Google Cluster Trace, single run)

| Model | Total Train Time | Per-epoch | Inference | GPU (train) | GPU (infer) |
|-------|-----------------|-----------|-----------|-------------|-------------|
| **T-LLM** | 45.7 min | 177 s | **0.22 ms/sample** | **1.7 GB** | **1.3 GB** |
| **T-LLM + Noisy** | **44.6 min** | 173 s | 0.23 ms/sample | 1.7 GB | 1.3 GB |
| Time-LLM | 113 min | 408 s | 0.71 ms/sample | 5.5 GB | 1.7 GB |

On Trace, T-LLM trains **2.5× faster** and runs inference **3.2× faster** than Time-LLM, using **3.3× less GPU memory**.

---

## Models

### T-LLM
**Paper**: [T-LLM: Teaching Large Language Models to Forecast Time Series via Temporal Distillation](https://arxiv.org/abs/2602.01937)

Key components:
- **RevIN**: per-sample instance normalization (mean/std stored for denormalization)
- **Input Block**: channel embedding → channel self-attention (teacher tokens) → cross-attention with compact GPT-2 word dictionary (student tokens)
- **Temporal Teacher**: 2-layer [DLinear trend + Adaptive Spectral Block + Trend-Periodic Fusion] (training only)
- **GPT-2 Student**: first 6 GPT-2 layers + LoRA (rank=8, alpha=16); backbone frozen
- **Distillation Loss**: `L = L_teach + 1.0·L_imit + 0.01·L_guide + 1.0·L_stud`
- **ClassificationHead** (Trace only): 8 hand-crafted memory statistics (trend, raw_mean, raw_std, raw_max, …) → 337-param MLP; uses `x_raw` (pre-RevIN) to preserve absolute level signal

Trainable params: ~12.5M (LoRA + teacher + input block + cls_head)

#### T-LLM + Noisy Teacher
Adds Gaussian noise to teacher intermediate features during distillation guidance (`--noise-std`).
Three adaptive strategies implemented:
- `--noise-decay`: multiplicative annealing per epoch (e.g. 0.95)
- `--noise-adaptive`: scales σ inversely with horizon — `σ_eff = σ × √(96/h)`
- `--noise-early-only`: noise on early features only; late features kept clean for long-range signal

### Time-LLM
**Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)

Key components:
- **Patch Embedding**: sliding window patches (patch_len=16, stride=8) → d_model=32
- **Reprogramming Layer**: cross-attention (Q=patches, K/V=learnable source embeddings mapped from GPT-2 vocabulary)
- **Prompt-as-Prefix**: per-sample statistics (min/max/median/trend/top-5 lags) + dataset description
- **Frozen GPT-2**: full 12-layer backbone, no gradient
- **RevIN**: per-sample instance normalization
- **ClassificationHead** (Trace only): same shared 337-param stats head as T-LLM, enabling an apples-to-apples completion-prediction comparison

Trainable params: ~52–58M (mapping layer ~50M dominates)

---

## Datasets

### ETTh1
Electricity Transformer Temperature dataset (hourly), 7 channels.
Auto-downloaded on first run if `--csv` is not found.

### Google Cluster Trace 2019
BigQuery extract of Google datacenter workloads. Each row is a 5-minute measurement.

| Column | Description |
|--------|-------------|
| `collection_id` | Job identifier |
| `instance_index` | Task within job |
| `date` | Timestamp |
| `memory` | Normalized memory usage (regression target, ch0) |
| `label` | 1 = completed, 0 = evicted/killed (classification target, ch1) |

**Split**: stratified instance-level (80/10/10 train/val/test by job ID, class ratio preserved).
Reproducible via `data/google-cluster/split_stratified.json`.

```
Instances: 3,779  (completed 50.6% / evicted 49.4%)
Train windows: 240,532  |  Val: 24,308  |  Test: 19,119
(late_ratio=0.5: windows sampled from final 50% of each job's timeline)
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Training

### T-LLM on ETTh1

```bash
python scripts/train_etth1.py --device cuda
```

With Noisy Teacher:
```bash
python scripts/train_etth1.py --device cuda \
    --noise-std 0.1 --noise-decay 0.95 --noise-adaptive --noise-early-only \
    --out results/etth1_noisy.json --ckpt-dir checkpoints/noisy
```

Key arguments:
```
--horizon         96 192 336 720   (default: all four)
--context-length  96
--epochs          50
--lr              5e-4
--patience        10
--noise-std       0.0              (>0 enables Noisy Teacher)
--noise-decay     1.0              (multiplicative decay per epoch)
--noise-adaptive                   (scale noise by sqrt(96/h))
--noise-early-only                 (noise on early features only)
--ckpt-dir        checkpoints
--out             results/etth1.json
```

### Time-LLM on ETTh1

```bash
python scripts/train_etth1_time_llm.py --device cuda --context-length 96
```

### T-LLM on Google Cluster Trace (multi-task)

```bash
python scripts/train_trace.py --device cuda \
    --csv data/google-cluster/cluster_trace.csv \
    --ckpt-dir checkpoints/tllm_trace
```

With Noisy Teacher:
```bash
python scripts/train_trace.py --device cuda --noise-std 0.1 \
    --ckpt-dir checkpoints/tllm_noisy_trace --out results/trace_noisy.json
```

### Time-LLM on Google Cluster Trace

```bash
python scripts/train_trace_time_llm.py --device cuda \
    --ckpt-dir checkpoints/timellm_trace
```

---

## Visualization

```bash
# T-LLM (basic)
python scripts/visualize_trace.py --device cuda \
    --ckpt checkpoints/tllm_trace/trace.pt

# T-LLM + Noisy
python scripts/visualize_trace.py --device cuda \
    --ckpt checkpoints/tllm_noisy_trace/trace.pt

# Time-LLM
python scripts/visualize_trace.py --device cuda \
    --ckpt checkpoints/timellm_trace/trace_time_llm.pt
```

Generates two plots per model in `results/plots/`:
- `trace_forecast_{model}.png` — memory prediction curves for 8 diverse jobs (4 completed / 4 evicted), with completion probability gauge
- `trace_cls_result_{model}.png` — ROC curve + confusion matrix over the full test set

---

## Project Structure

```
t_llm/
  config.py           TLLMConfig dataclass
  data.py             ETTh1 dataset / loader
  data_trace.py       Google Cluster Trace dataset (stratified split)
  layers.py           RevIN, DLinear, SpectralBlock, ForecastHead, …
  losses.py           DistillationLoss (+ Noisy Teacher noise injection)
  model.py            TLLM, TemporalTeacher, GPT2LoRAStudent, ClassificationHead

time_llm/
  config.py           TimeLLMConfig dataclass
  model.py            TimeLLM (reprogramming + frozen GPT-2)

scripts/
  train_etth1.py              T-LLM training on ETTh1
  train_etth1_time_llm.py     Time-LLM training on ETTh1
  train_trace.py              T-LLM multi-task training on Trace
  train_trace_time_llm.py     Time-LLM training on Trace
  visualize_trace.py          Trace prediction + classification visualization

data/
  ETT-small/ETTh1.csv                   auto-downloaded
  google-cluster/cluster_trace.csv      BigQuery export
  google-cluster/split_stratified.json  reproducible train/val/test split

results/
  etth1.json              T-LLM ETTh1 results
  etth1_noisy_v2.json     T-LLM + Noisy ETTh1 results
  etth1_time_llm.json     Time-LLM ETTh1 results
  trace_tllm.json         T-LLM Trace results
  trace_noisy.json        T-LLM + Noisy Trace results
  trace_time_llm.json     Time-LLM Trace results
  plots/                  visualization outputs

checkpoints/              saved model weights (.pt)
seraph/                   SLURM job scripts for cluster execution
```

---

## Notes

- **Classification is driven by raw statistics, not the LLM backbone**: All three models share the same 337-param stats head computed from pre-RevIN memory (`x_raw`). This head is fully decoupled from the forecasting path, so T-LLM, T-LLM+Noisy, and Time-LLM reach near-identical completion accuracy (0.71 / 0.71 / 0.69). The takeaway: **job completion is predictable from absolute memory level/trend, regardless of which forecasting model is attached.**
- **Why the label channel needs a separate head**: The label is constant per job (all 0s or all 1s). RevIN normalizes this to all-zeros regardless of the true class, making the two classes indistinguishable if predicted as a forecasting channel. The stats head sidesteps this by reading `x_raw` directly.
- **Noisy Teacher does not significantly improve T-LLM on Trace**: Noise affects only the guidance loss (`L_guide`, λ=0.01), which contributes minimally to the total loss dominated by BCE (λ=5), and the classification head is decoupled from the teacher path entirely.
- **ETTh1 teacher overfitting**: Teacher val MSE consistently degrades after epoch 2–3 across all horizons. The student checkpoint is selected at this early point, limiting the benefit of longer training or noise strategies.
- **val_mse vs test_mse gap on Trace** (0.022 vs 0.17): val jobs have narrower memory ranges; test metrics are the ground truth for paper reporting.
