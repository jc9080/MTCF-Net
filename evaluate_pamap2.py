"""Evaluate a trained MTCF-Net checkpoint on PAMAP2."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pamap2 import create_pamap2_loaders
from models.mtcf_net import mtcf_net_pamap2
from utils.checkpoint import load_best_model
from utils.metrics import MetricAccumulator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate MTCF-Net on PAMAP2."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/PAMAP2/PAMAP2_Dataset",
        help="Path to the extracted PAMAP2 dataset.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pamap2_test",
        help="Directory containing best_model.pth.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model loss, accuracy, and macro F1."""

    model.eval()

    metrics = MetricAccumulator(num_classes=18)
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        data_loader,
        desc="Evaluating",
        leave=False,
    )

    for inputs, targets in progress:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        metrics.update(
            logits.detach(),
            targets.detach(),
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    if total_samples == 0:
        raise RuntimeError("The evaluation DataLoader is empty.")

    average_loss = total_loss / total_samples
    computed_metrics = metrics.compute()

    return (
        average_loss,
        computed_metrics.accuracy,
        computed_metrics.macro_f1,
    )


def main() -> None:
    """Load the best checkpoint and evaluate it."""

    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("MTCF-Net PAMAP2 Evaluation")
    print("=" * 60)
    print(f"Device          : {device}")
    print(f"Data directory  : {args.data_dir}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Batch size      : {args.batch_size}")

    if torch.cuda.is_available():
        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    loaders = create_pamap2_loaders(
	data_dir=args.data_dir,
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	)

    if not isinstance(loaders, (tuple, list)) or len(loaders) < 3:
        raise RuntimeError(
            "create_pamap2_loaders() must return at least three loaders."
        )

    test_loader = loaders[2]

    model = mtcf_net_pamap2(num_classes=18)
    model = model.to(device)

    load_best_model(
        model=model,
        save_dir=checkpoint_dir,
        map_location=device,
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy, test_macro_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"Test Macro F1 : {test_macro_f1 * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
