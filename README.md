# T-LLM vs Time-LLM — ETTh1 Benchmark

PyTorch reimplementation of two LLM-based time series forecasting methods, evaluated on ETTh1 long-term forecasting.

| Model | Backbone | h=96 | h=192 | h=336 | h=720 | Avg MSE |
|-------|----------|------|-------|-------|-------|---------|
| **T-LLM** (ours) | GPT-2 + LoRA | 0.3916 | 0.4496 | 0.4862 | 0.5148 | **0.4605** |
| **Time-LLM** (ours) | GPT-2 frozen | 0.4060 | 0.4599 | 0.4895 | 0.5156 | 0.4678 |
| T-LLM (paper) | GPT-2 + LoRA | 0.417 | 0.439 | 0.458 | 0.449 | 0.441 |
| Time-LLM (paper) | LLaMA-7B | 0.404 | 0.448 | 0.481 | 0.461 | 0.449 |

> Paper results for Time-LLM use LLaMA-7B. Both our runs use GPT-2 for a fair backbone-controlled comparison.

---

## Models

### T-LLM
**Paper**: [T-LLM: Teaching Large Language Models to Forecast Time Series via Temporal Distillation](https://arxiv.org/abs/2602.01937)

Key components:
- **Input Block**: channel embedding → channel self-attention (teacher tokens) → cross-attention with compact GPT-2 word dictionary (student tokens)
- **Temporal Teacher**: 2-layer DLinear trend block + Adaptive Spectral Block + Trend-Periodic Fusion gate (training only)
- **GPT-2 Student**: first 6 GPT-2 layers + LoRA (rank=8, alpha=16)
- **Distillation Loss**: L = L_teach + 1.0·L_imit + 0.01·L_guide + 1.0·L_stud
- **RevIN**: per-sample instance normalization

Trainable params: ~10.4M (LoRA adapters + teacher + input block)

### Time-LLM
**Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)

Key components:
- **Patch Embedding**: sliding window patches (patch_len=16, stride=8) → d_model=32
- **Mapping Layer**: learnable projection over GPT-2 vocabulary → source embeddings (K/V)
- **Reprogramming Layer**: multi-head cross-attention (Q=patches, K/V=source embeddings)
- **Prompt-as-Prefix**: per-sample statistics (min/max/median/trend/top-5 lags) + dataset/task description prepended to LLM input
- **Frozen GPT-2**: full 12-layer backbone, no gradient
- **RevIN**: per-sample instance normalization

Trainable params: ~56.8M (mapping_layer dominates at ~50M)

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Training

### T-LLM on ETTh1

```bash
python scripts/train_etth1.py --device cuda --csv data/ETT-small/ETTh1.csv
```

Key arguments:
```
--horizon       96 192 336 720   (default: all four)
--context-length 96              (paper default)
--epochs        50
--batch-size    64
--lr            5e-4
--min-epochs    3
--patience      10
--ckpt-dir      checkpoints
--out           results/etth1.json
```

### Time-LLM on ETTh1

```bash
python scripts/train_etth1_time_llm.py --device cuda --csv data/ETT-small/ETTh1.csv
```

Key arguments:
```
--horizon         96 192 336 720   (default: all four)
--context-length  512              (paper default)
--epochs          100
--batch-size      24
--lr              0.01
--patience        20
--ckpt-dir        checkpoints
--out             results/etth1_time_llm.json
```

ETTh1 data is downloaded automatically on first run of `train_etth1.py` if not found at `--csv`.

---

## Results

Results are saved as JSON to `results/`:
- `results/etth1.json` — T-LLM per-horizon results
- `results/etth1_time_llm.json` — Time-LLM per-horizon results

---

## Input / Output Format

```
Input:  [batch, context_length, channels]
Output: [batch, prediction_length, channels]
```

---

## Project Structure

```
t_llm/          T-LLM package (config, data, layers, model, losses)
time_llm/       Time-LLM package (config + model)
scripts/
  train_etth1.py            T-LLM training
  train_etth1_time_llm.py   Time-LLM training
data/ETT-small/ETTh1.csv    auto-downloaded
results/                    JSON result files
checkpoints/                saved model weights
```
