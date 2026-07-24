from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def _load_signal_file(file_path: Path) -> np.ndarray:
    """Load one UCI-HAR inertial signal file.

    Returns:
        Array with shape (num_samples, sequence_length).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    data = np.loadtxt(file_path, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D signal array, but got shape {data.shape}: {file_path}"
        )

    return data


def load_uci_har_split(
    root_dir: str | Path,
    split: Literal["train", "test"],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one UCI-HAR split.

    Args:
        root_dir:
            Path to the extracted 'UCI HAR Dataset' directory.
        split:
            Either 'train' or 'test'.

    Returns:
        x:
            Sensor sequence array with shape
            (num_samples, sequence_length, num_channels).
        y:
            Zero-based labels with shape (num_samples,).
    """
    root_dir = Path(root_dir).expanduser().resolve()
    split_dir = root_dir / split
    signal_dir = split_dir / "Inertial Signals"

    signals = []

    for signal_name in SIGNAL_NAMES:
        file_path = signal_dir / f"{signal_name}_{split}.txt"
        signal = _load_signal_file(file_path)
        signals.append(signal)

    # Six arrays shaped (N, T) -> one array shaped (N, T, C)
    x = np.stack(signals, axis=-1).astype(np.float32)

    label_path = split_dir / f"y_{split}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Original UCI-HAR labels are 1 to 6.
    # PyTorch CrossEntropyLoss expects labels from 0 to 5.
    y = np.loadtxt(label_path, dtype=np.int64).reshape(-1) - 1

    if len(x) != len(y):
        raise ValueError(
            f"Sample-label count mismatch: x={len(x)}, y={len(y)}"
        )

    if x.shape[1] != 128:
        raise ValueError(
            f"Expected sequence length 128, but got {x.shape[1]}"
        )

    if x.shape[2] != len(SIGNAL_NAMES):
        raise ValueError(
            f"Expected {len(SIGNAL_NAMES)} channels, but got {x.shape[2]}"
        )

    if y.min() < 0 or y.max() > 5:
        raise ValueError(
            f"Expected zero-based labels in [0, 5], but got "
            f"min={y.min()}, max={y.max()}"
        )

    return x, y


class UCIHARDataset(Dataset):
    """UCI-HAR dataset returning (sequence, label)."""

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "test"],
        random_mask_ratio: float = 0.0,
        gaussian_noise_std: float = 0.0,
    ) -> None:
        if not 0.0 <= random_mask_ratio <= 1.0:
            raise ValueError("random_mask_ratio must be between 0 and 1.")

        if gaussian_noise_std < 0.0:
            raise ValueError("gaussian_noise_std must be non-negative.")

        x, y = load_uci_har_split(root_dir, split)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        self.split = split
        self.random_mask_ratio = random_mask_ratio
        self.gaussian_noise_std = gaussian_noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index].clone()
        y = self.y[index]

        # Apply augmentation only to training samples.
        if self.split == "train":
            if self.random_mask_ratio > 0.0:
                time_mask = torch.rand(x.shape[0]) < self.random_mask_ratio
                x[time_mask, :] = 0.0

            if self.gaussian_noise_std > 0.0:
                noise = torch.randn_like(x) * self.gaussian_noise_std
                x = x + noise

        return x, y


def create_uci_har_loaders(
    root_dir: str | Path,
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    num_workers: int = 4,
    random_mask_ratio: float = 0.04,
    gaussian_noise_std: float = 0.01,
) -> tuple[DataLoader, DataLoader]:
    """Create training and test DataLoaders."""

    train_dataset = UCIHARDataset(
        root_dir=root_dir,
        split="train",
        random_mask_ratio=random_mask_ratio,
        gaussian_noise_std=gaussian_noise_std,
    )

    test_dataset = UCIHARDataset(
        root_dir=root_dir,
        split="test",
        random_mask_ratio=0.0,
        gaussian_noise_std=0.0,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[1] / "data" / "UCI HAR Dataset"

    train_loader, test_loader = create_uci_har_loaders(
        root_dir=default_root,
        train_batch_size=128,
        test_batch_size=128,
        num_workers=0,
    )

    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))

    print("Train dataset size:", len(train_loader.dataset))
    print("Test dataset size:", len(test_loader.dataset))
    print("Train batch x:", train_x.shape, train_x.dtype)
    print("Train batch y:", train_y.shape, train_y.dtype)
    print("Test batch x:", test_x.shape, test_x.dtype)
    print("Test batch y:", test_y.shape, test_y.dtype)
    print("Train label range:", train_y.min().item(), train_y.max().item())
