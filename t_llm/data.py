from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, channels]")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        values = values.astype("float32")
        if normalize:
            self.mean = values.mean(axis=0, keepdims=True)
            self.std = values.std(axis=0, keepdims=True) + 1e-6
            values = (values - self.mean) / self.std
        else:
            self.mean = np.zeros((1, values.shape[1]), dtype="float32")
            self.std = np.ones((1, values.shape[1]), dtype="float32")
        self.values = values
        self.length = max(0, (len(values) - context_length - prediction_length) // stride + 1)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        context_length: int,
        prediction_length: int,
        target_columns: list[str] | None = None,
        stride: int = 1,
    ) -> "SlidingWindowDataset":
        frame = pd.read_csv(path)
        if target_columns is None:
            numeric = frame.select_dtypes(include=["number"])
        else:
            numeric = frame[target_columns]
        return cls(numeric.to_numpy(), context_length, prediction_length, stride=stride)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        mid = start + self.context_length
        end = mid + self.prediction_length
        return torch.from_numpy(self.values[start:mid]), torch.from_numpy(self.values[mid:end])


class WindowedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        context_length: int,
        prediction_length: int,
        window_start: int,
        window_end: int,
        stride: int = 1,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, channels]")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.window_start = window_start
        self.window_end = window_end
        values = values.astype("float32")
        self.mean = mean.astype("float32") if mean is not None else values.mean(axis=0, keepdims=True)
        self.std = std.astype("float32") if std is not None else values.std(axis=0, keepdims=True) + 1e-6
        self.values = (values - self.mean) / self.std
        max_start = window_end - context_length - prediction_length
        self.length = max(0, (max_start - window_start) // stride + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.window_start + index * self.stride
        mid = start + self.context_length
        end = mid + self.prediction_length
        return torch.from_numpy(self.values[start:mid]), torch.from_numpy(self.values[mid:end])


def load_numeric_csv(path: str | Path, target_columns: list[str] | None = None) -> np.ndarray:
    frame = pd.read_csv(path)
    if target_columns is None:
        numeric = frame.select_dtypes(include=["number"])
    else:
        numeric = frame[target_columns]
    return numeric.to_numpy(dtype="float32")


def make_ett_hour_datasets(
    path: str | Path,
    context_length: int,
    prediction_length: int,
    target_columns: list[str] | None = None,
    stride: int = 1,
) -> tuple[WindowedTimeSeriesDataset, WindowedTimeSeriesDataset, WindowedTimeSeriesDataset]:
    values = load_numeric_csv(path, target_columns)
    train_end = 12 * 30 * 24
    valid_end = train_end + 4 * 30 * 24
    test_end = min(len(values), valid_end + 4 * 30 * 24)

    mean = values[:train_end].mean(axis=0, keepdims=True)
    std = values[:train_end].std(axis=0, keepdims=True) + 1e-6
    train = WindowedTimeSeriesDataset(values, context_length, prediction_length, 0, train_end, stride, mean, std)
    valid = WindowedTimeSeriesDataset(
        values,
        context_length,
        prediction_length,
        train_end - context_length,
        valid_end,
        stride,
        mean,
        std,
    )
    test = WindowedTimeSeriesDataset(
        values,
        context_length,
        prediction_length,
        valid_end - context_length,
        test_end,
        stride,
        mean,
        std,
    )
    return train, valid, test
