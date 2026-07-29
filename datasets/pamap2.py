from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ACTIVITY_MAP = {
    1: 0,    # lying
    2: 1,    # sitting
    3: 2,    # standing
    4: 3,    # walking
    5: 4,    # running
    6: 5,    # cycling
    7: 6,    # Nordic walking
    9: 7,    # watching TV
    10: 8,   # computer work
    11: 9,   # car driving
    12: 10,  # ascending stairs
    13: 11,  # descending stairs
    16: 12,  # vacuum cleaning
    17: 13,  # ironing
    18: 14,  # folding laundry
    19: 15,  # house cleaning
    20: 16,  # playing soccer
    24: 17,  # rope jumping
}


# 54개 원본 열 중 사용할 28개 입력 채널
# 2: heart rate
# 각 IMU에서 ±16g accelerometer 3축, gyroscope 3축,
# magnetometer 3축 사용
FEATURE_COLUMNS = [
    2,

    # Hand IMU
    4, 5, 6,
    10, 11, 12,
    13, 14, 15,

    # Chest IMU
    21, 22, 23,
    27, 28, 29,
    30, 31, 32,

    # Ankle IMU
    38, 39, 40,
    44, 45, 46,
    47, 48, 49,
]


class PAMAP2Dataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        mask_ratio: float = 0.04,
        noise_std: float = 0.01,
    ) -> None:
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.windows[index].clone()
        y = self.labels[index]

        if self.augment:
            if self.mask_ratio > 0:
                mask = torch.rand(x.shape[0]) < self.mask_ratio
                x[mask] = 0.0

            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std

        return x, y


def _interpolate_nan(features: np.ndarray) -> np.ndarray:
    features = features.copy()

    for column_idx in range(features.shape[1]):
        column = features[:, column_idx]
        valid = ~np.isnan(column)

        if not valid.any():
            features[:, column_idx] = 0.0
            continue

        indices = np.arange(len(column))
        features[:, column_idx] = np.interp(
            indices,
            indices[valid],
            column[valid],
        )

    return features


def _load_subject_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path, dtype=np.float32)

    activity_ids = data[:, 1].astype(np.int64)
    valid_rows = np.isin(activity_ids, list(ACTIVITY_MAP.keys()))

    data = data[valid_rows]
    activity_ids = activity_ids[valid_rows]

    features = data[:, FEATURE_COLUMNS]
    features = _interpolate_nan(features)

    labels = np.array(
        [ACTIVITY_MAP[int(activity_id)] for activity_id in activity_ids],
        dtype=np.int64,
    )

    return features, labels


def _create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
) -> tuple[list[np.ndarray], list[int]]:
    windows: list[np.ndarray] = []
    window_labels: list[int] = []

    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size

        x_window = features[start:end]
        y_window = labels[start:end]

        values, counts = np.unique(y_window, return_counts=True)
        majority_label = int(values[np.argmax(counts)])

        windows.append(x_window)
        window_labels.append(majority_label)

    return windows, window_labels


def _build_split(
    root: Path,
    subject_ids: Iterable[int],
    include_optional: bool = True,
    window_size: int = 256,
    stride: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []

    folders = ["Protocol"]
    if include_optional:
        folders.append("Optional")

    for subject_id in subject_ids:
        for folder in folders:
            file_path = root / folder / f"subject{subject_id}.dat"

            if not file_path.exists():
                continue

            features, labels = _load_subject_file(file_path)
            windows, window_labels = _create_windows(
                features,
                labels,
                window_size=window_size,
                stride=stride,
            )

            all_windows.extend(windows)
            all_labels.extend(window_labels)

    if not all_windows:
        raise RuntimeError(
            f"No PAMAP2 windows were created for subjects: {list(subject_ids)}"
        )

    return (
        np.asarray(all_windows, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
    )


def _normalize_splits(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, val_x, test_x


def create_pamap2_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    window_size: int = 256,
    stride: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(data_dir)

    # Subject-independent split
    train_subjects = [101, 102, 103, 104, 105, 106]
    val_subjects = [107]
    test_subjects = [108, 109]

    train_x, train_y = _build_split(
        root,
        train_subjects,
        include_optional=True,
        window_size=window_size,
        stride=stride,
    )

    val_x, val_y = _build_split(
        root,
        val_subjects,
        include_optional=True,
        window_size=window_size,
        stride=stride,
    )

    test_x, test_y = _build_split(
        root,
        test_subjects,
        include_optional=True,
        window_size=window_size,
        stride=stride,
    )

    train_x, val_x, test_x = _normalize_splits(
        train_x,
        val_x,
        test_x,
    )

    train_dataset = PAMAP2Dataset(
        train_x,
        train_y,
        augment=True,
    )
    val_dataset = PAMAP2Dataset(
        val_x,
        val_y,
        augment=False,
    )
    test_dataset = PAMAP2Dataset(
        test_x,
        test_y,
        augment=False,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(
        f"PAMAP2 train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print(
        f"Input shape={train_dataset.windows.shape}, "
        f"num_classes={len(ACTIVITY_MAP)}"
    )

    return train_loader, val_loader, test_loader
