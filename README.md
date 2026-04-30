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

## ETTh1 Paper Protocol On CUDA / WSL

The paper's long-term ETTh1 setting uses an input length of 96 and prediction
horizons of 96, 192, 336, and 720. The reproduction script downloads ETTh1 when
`data/ETT-small/ETTh1.csv` is missing, trains the temporal teacher and GPT-2
LoRA student together, and evaluates with the student branch only. In the GPT-2
path, the input block builds the paper-style compact word-embedding dictionary
from the frozen GPT-2 embeddings and uses cross attention to produce the LLM
student tokens separately from the temporal teacher tokens.
The default optimizer is Adam with learning rate `0.0005`, and the loss weights
follow the paper's `lambda1=1.0` for prediction imitation, `lambda2=0.01` for
temporal guidance, and `lambda3=1.0` for student supervision.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_etth1_paper_protocol.py --device cuda --amp
```

For a quick CUDA smoke run before the full experiment:

```bash
python scripts/run_etth1_paper_protocol.py --device cuda --horizons 96 --epochs 1 --batch-size 16 --amp
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
