"""
Google Cluster Trace 데이터 로더.

CSV 컬럼 (BigQuery 추출 결과):
    collection_id, instance_index, date,
    cpu, memory, max_cpu, max_memory, avg_cpi,
    label (1=완료, 0=중단),
    duration_sec

채널 구성 (2채널):
    ch 0 : memory  → StandardScaler 정규화 (regression target)
    ch 1 : label   → 0/1 그대로 (classification target)

Instance-level split:
    전체 job을 무작위로 train/val/test로 나눔.
    시간 기준 분리가 아니므로 각 job의 전체 이력이 한 split에 들어감.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    """
    collection_id × instance_index 단위로 시계열을 구성.
    context_length 만큼 입력, prediction_length 만큼 예측.

    반환: (x, y)
        x: (context_length, 2)  — [memory, label]
        y: (prediction_length, 2) — [memory, label]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        instance_ids: list[tuple],   # (collection_id, instance_index) 목록
        context_length: int,
        prediction_length: int,
        scaler: StandardScaler,
        fit_scaler: bool = False,
    ) -> None:
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.scaler            = scaler

        # instance별로 시계열 슬라이딩 윈도우 생성
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        all_memory = []

        for cid, iidx in instance_ids:
            ts = (
                df[(df["collection_id"] == cid) & (df["instance_index"] == iidx)]
                .sort_values("date")[["memory", "label"]]
                .values.astype("float32")
            )
            if fit_scaler:
                all_memory.append(ts[:, 0])

            total = context_length + prediction_length
            if len(ts) < total:
                continue
            for start in range(0, len(ts) - total + 1):
                x = ts[start            : start + context_length]
                y = ts[start + context_length : start + total]
                self.samples.append((x, y))

        if fit_scaler and all_memory:
            flat = np.concatenate(all_memory).reshape(-1, 1)
            self.scaler.fit(flat)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        """ch0(memory)만 정규화, ch1(label)은 그대로."""
        out = arr.copy()
        out[:, 0:1] = self.scaler.transform(arr[:, 0:1])
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        import torch
        return (
            torch.tensor(self._scale(x), dtype=torch.float32),
            torch.tensor(self._scale(y), dtype=torch.float32),
        )


def load_trace(
    csv_path: str | Path,
    context_length: int = 24,
    prediction_length: int = 12,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> tuple[TraceDataset, TraceDataset, TraceDataset]:
    """
    CSV를 읽어 instance-level split 후 (train, val, test) 반환.

    split_file: 지정하면 split 결과를 JSON으로 저장/재현.
    """
    df = pd.read_csv(csv_path)

    # 필수 컬럼 확인
    required = {"collection_id", "instance_index", "date", "memory", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 컬럼 없음: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    # 유효한 인스턴스 (context + prediction 길이 이상) 선별
    min_len = context_length + prediction_length
    instance_ids = [
        (cid, iidx)
        for (cid, iidx), grp in df.groupby(["collection_id", "instance_index"])
        if len(grp) >= min_len
    ]
    print(f"[TraceDataset] 유효 인스턴스: {len(instance_ids):,}개  "
          f"(최소 {min_len} 스텝 이상)")

    # split 로드 또는 생성
    split_path = Path(split_file) if split_file else None
    if split_path and split_path.exists() and split_path.stat().st_size > 0:
        with open(split_path) as f:
            splits = json.load(f)
        train_ids = [tuple(x) for x in splits["train"]]
        val_ids   = [tuple(x) for x in splits["val"]]
        test_ids  = [tuple(x) for x in splits["test"]]
        print(f"  split 재사용: {split_file}")
    else:
        rng = random.Random(seed)
        shuffled = instance_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train : n_train + n_val]
        test_ids  = shuffled[n_train + n_val:]
        if split_file:
            Path(split_file).parent.mkdir(parents=True, exist_ok=True)
            # numpy.int64 → Python int 변환 후 직렬화
            def to_py(ids):
                return [[int(c), int(i)] for c, i in ids]
            with open(split_file, "w") as f:
                json.dump({"train": to_py(train_ids),
                           "val":   to_py(val_ids),
                           "test":  to_py(test_ids)}, f)
            print(f"  split 저장: {split_file}")

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    scaler = StandardScaler()
    train_set = TraceDataset(df, train_ids, context_length, prediction_length,
                             scaler, fit_scaler=True)
    val_set   = TraceDataset(df, val_ids,   context_length, prediction_length, scaler)
    test_set  = TraceDataset(df, test_ids,  context_length, prediction_length, scaler)
    print(f"  샘플 수 — train={len(train_set):,}  "
          f"val={len(val_set):,}  test={len(test_set):,}")
    return train_set, val_set, test_set
