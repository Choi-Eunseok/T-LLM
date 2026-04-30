# T-LLM

PyTorch implementation scaffold for **T-LLM: Teaching Large Language Models to Forecast Time Series via Temporal Distillation**.

This repo implements the paper's main training idea:

- shared time-series input block
- training-only temporal teacher with DLinear-style trend modeling and FFT spectral modeling
- student forecasting branch
- reverse distillation losses: teacher supervision, student supervision, prediction imitation, and head/tail temporal guidance
- inference through the student branch only

The default student is a compact Transformer so the project can run without downloading an LLM. The `TLLM` class keeps the student boundary isolated, so a frozen GPT-style backbone plus LoRA adapter can be added later without changing the teacher or loss code.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Shape Check

```bash
python scripts/smoke_train.py
```

## Train On A CSV

```bash
python scripts/train_csv.py --csv path/to/data.csv --context-length 96 --prediction-length 96
```

To forecast only selected numeric columns:

```bash
python scripts/train_csv.py --csv path/to/data.csv --columns OT HUFL HULL
```

## Input Format

Model input is a float tensor with shape:

```text
[batch, context_length, channels]
```

Forecast output is:

```text
[batch, prediction_length, channels]
```
