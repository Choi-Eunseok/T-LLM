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
        late_ratio: float = 0.5,     # job 후반 몇 % 구간에서 윈도우 생성
                                     # 0.0 = 전체, 0.5 = 후반 50%
    ) -> None:
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.scaler            = scaler

        # instance별로 시계열 슬라이딩 윈도우 생성
        self.samples:  list[tuple[np.ndarray, np.ndarray]] = []
        self.job_ids:  list[tuple] = []   # 각 샘플의 (collection_id, instance_index)
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

            # late_ratio > 0 이면 job 후반부에서만 윈도우 생성
            # 종료 직전 패턴(완료 vs 중단)이 구별 가능한 구간
            start_min = max(0, int(len(ts) * late_ratio) - context_length)
            for start in range(start_min, len(ts) - total + 1):
                x = ts[start                  : start + context_length]
                y = ts[start + context_length : start + total]
                self.samples.append((x, y))
                self.job_ids.append((cid, iidx))

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
    late_ratio: float = 0.5,
) -> tuple[TraceDataset, TraceDataset, TraceDataset]:
    """
    CSV를 읽어 stratified instance-level split 후 (train, val, test) 반환.

    split_file: 지정하면 split 결과를 JSON으로 저장/재현.

    Stratified split 이유:
      무작위 split 시 val/test의 class 비율이 크게 달라질 수 있다.
      (예: val=55% completed, test=40% completed)
      → val 기준으로 학습한 모델이 test에서 역방향으로 틀림.
      완료(label=1) / 중단(label=0)을 각각 분리해 동일 비율로 분배한다.
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
        # ── Stratified split: completed / evicted 비율을 각 split에서 동일하게 유지 ──
        # job당 label은 모든 row에서 동일 (상수값). 첫 row에서 가져옴.
        job_label_map = (
            df.groupby(["collection_id", "instance_index"])["label"]
            .first()
            .to_dict()
        )
        pos_ids = [id for id in instance_ids if job_label_map.get(id, 0) == 1]
        neg_ids = [id for id in instance_ids if job_label_map.get(id, 0) == 0]

        rng = random.Random(seed)
        rng.shuffle(pos_ids)
        rng.shuffle(neg_ids)

        def strat_split(ids):
            n       = len(ids)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

        pos_tr, pos_va, pos_te = strat_split(pos_ids)
        neg_tr, neg_va, neg_te = strat_split(neg_ids)

        train_ids = pos_tr + neg_tr
        val_ids   = pos_va + neg_va
        test_ids  = pos_te + neg_te

        # 클래스 분포 출력 (검증용)
        n_pos_val  = len(pos_va);  n_pos_te = len(pos_te)
        n_neg_val  = len(neg_va);  n_neg_te = len(neg_te)
        print(f"  클래스 비율 — "
              f"completed(1): {len(pos_ids)}/{len(instance_ids)} "
              f"({100*len(pos_ids)/len(instance_ids):.1f}%)  "
              f"evicted(0): {len(neg_ids)}/{len(instance_ids)} "
              f"({100*len(neg_ids)/len(instance_ids):.1f}%)")
        print(f"  val  — completed={n_pos_val}, evicted={n_neg_val} "
              f"(pos_ratio={100*n_pos_val/max(n_pos_val+n_neg_val,1):.1f}%)")
        print(f"  test — completed={n_pos_te}, evicted={n_neg_te} "
              f"(pos_ratio={100*n_pos_te/max(n_pos_te+n_neg_te,1):.1f}%)")

        if split_file:
            Path(split_file).parent.mkdir(parents=True, exist_ok=True)
            def to_py(ids):
                return [[int(c), int(i)] for c, i in ids]
            with open(split_file, "w") as f:
                json.dump({"train": to_py(train_ids),
                           "val":   to_py(val_ids),
                           "test":  to_py(test_ids)}, f)
            print(f"  split 저장 (stratified): {split_file}")

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    scaler = StandardScaler()
    train_set = TraceDataset(df, train_ids, context_length, prediction_length,
                             scaler, fit_scaler=True, late_ratio=late_ratio)
    val_set   = TraceDataset(df, val_ids,   context_length, prediction_length,
                             scaler, late_ratio=late_ratio)
    test_set  = TraceDataset(df, test_ids,  context_length, prediction_length,
                             scaler, late_ratio=late_ratio)
    print(f"  샘플 수 — train={len(train_set):,}  "
          f"val={len(val_set):,}  test={len(test_set):,}  "
          f"(late_ratio={late_ratio})")
    return train_set, val_set, test_set
