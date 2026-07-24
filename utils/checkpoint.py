from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
) -> None:
    """
    Save a complete training checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    """
    Load checkpoint.

    Returns
    -------
    epoch, best_metric
    """
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return (
        checkpoint["epoch"],
        checkpoint["best_metric"],
    )


def save_best_model(
    model: nn.Module,
    save_dir: str | Path,
    metric: float,
    best_metric: float,
) -> float:
    """
    Save best model only if metric improves.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if metric > best_metric:

        torch.save(
            model.state_dict(),
            save_dir / "best_model.pth",
        )

        print(
            f"New best model: "
            f"{best_metric:.4f} → {metric:.4f}"
        )

        return metric

    return best_metric


def load_best_model(
    model: nn.Module,
    save_dir: str | Path,
    map_location="cpu",
):
    """
    Load best model weights only.
    """
    save_dir = Path(save_dir)

    model.load_state_dict(
        torch.load(
            save_dir / "best_model.pth",
            map_location=map_location,
        )
    )


if __name__ == "__main__":

    from models.mtcf_net import mtcf_net_uci_har

    model = mtcf_net_uci_har()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
    )

    save_checkpoint(
        "checkpoints/test_checkpoint.pth",
        model,
        optimizer,
        scheduler,
        epoch=3,
        best_metric=0.82,
    )

    epoch, metric = load_checkpoint(
        "checkpoints/test_checkpoint.pth",
        model,
        optimizer,
        scheduler,
    )

    print(epoch)
    print(metric)

    metric = save_best_model(
        model,
        "checkpoints",
        metric=0.85,
        best_metric=0.82,
    )

    print(metric)

    load_best_model(
        model,
        "checkpoints",
    )

    print("Checkpoint test passed.")
