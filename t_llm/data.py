from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ETThDataset(Dataset):
    """Sliding-window dataset over a pre-split ETT-hour segment."""

    def __init__(
        self,
        values: np.ndarray,
        context_length: int,
        prediction_length: int,
        start: int,
        end: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.context_length = context_length
        self.prediction_length = prediction_length
        normalized = (values.astype("float32") - mean) / std
        # crop to window, keeping a lead of context_length for the first sample
        self.data = normalized[start:end]
        self.offset = start
        max_start = (end - start) - context_length - prediction_length
        self.length = max(0, max_start + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + self.context_length : idx + self.context_length + self.prediction_length]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_etth1(
    path: str | Path,
    context_length: int,
    prediction_length: int,
) -> tuple[ETThDataset, ETThDataset, ETThDataset]:
    """
    Load ETTh1 and return (train, val, test) datasets.

    Splits follow the standard ETT-hour protocol used in the paper:
      train : first 12 months  (12 * 30 * 24 = 8640 time steps)
      val   : next  4 months   (4  * 30 * 24 = 2880 time steps)
      test  : next  4 months   (4  * 30 * 24 = 2880 time steps)

    Normalisation statistics are computed on the training split only.
    Val/test windows overlap with the previous split by context_length
    so the first sample always has a full look-back window.
    """
    df = pd.read_csv(path)
    values = df.select_dtypes(include="number").to_numpy(dtype="float32")

    train_end = 12 * 30 * 24          # 8640
    val_end   = train_end + 4 * 30 * 24   # 11520
    test_end  = min(len(values), val_end + 4 * 30 * 24)  # 14400

    mean = values[:train_end].mean(axis=0, keepdims=True)
    std  = values[:train_end].std(axis=0, keepdims=True) + 1e-6

    train = ETThDataset(values, context_length, prediction_length, 0, train_end, mean, std)
    val   = ETThDataset(values, context_length, prediction_length,
                        train_end - context_length, val_end, mean, std)
    test  = ETThDataset(values, context_length, prediction_length,
                        val_end - context_length, test_end, mean, std)
    return train, val, test
