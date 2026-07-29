"""Training entry point for MTCF-Net on the PAMAP2 dataset."""

from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from datasets.pamap2 import create_pamap2_loaders
from engine.trainer import Trainer
from models.mtcf_net import (
    count_trainable_parameters,
    mtcf_net_pamap2,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train MTCF-Net on PAMAP2."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/PAMAP2/PAMAP2_Dataset",
        help="Path to the extracted PAMAP2 dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pamap2",
        help="Directory in which checkpoints are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path from which training resumes.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def supported_kwargs(
    function: Callable[..., Any],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    """Return candidate keyword arguments supported by a callable.

    This permits minor differences between the parameter names used by
    individual project modules.
    """

    signature = inspect.signature(function)

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if has_var_kwargs:
        return candidates

    return {
        name: value
        for name, value in candidates.items()
        if name in signature.parameters
    }


def create_loaders(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create the PAMAP2 training and validation/test loaders."""

    loader_candidates = {
        "data_dir": args.data_dir,
        "root": args.data_dir,
        "root_dir": args.data_dir,
        "dataset_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    kwargs = supported_kwargs(
        create_pamap2_loaders,
        loader_candidates,
    )

    print(f"DataLoader arguments: {kwargs}")

    loaders = create_pamap2_loaders(**kwargs)

    if not isinstance(loaders, (tuple, list)) or len(loaders) < 2:
        raise RuntimeError(
            "create_pamap2_loaders() must return at least two loaders."
        )

    train_loader = loaders[0]
    validation_loader = loaders[1]

    return train_loader, validation_loader


def create_model() -> nn.Module:
    """Create the PAMAP2 version of MTCF-Net."""

    model_candidates = {
        "num_classes": 18,
    }

    kwargs = supported_kwargs(
        mtcf_net_pamap2,
        model_candidates,
    )

    print(f"Model arguments: {kwargs}")

    model = mtcf_net_pamap2(**kwargs)

    if not isinstance(model, nn.Module):
        raise TypeError(
            "mtcf_net_pamap2() did not return torch.nn.Module."
        )

    return model


def create_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Trainer:
    """Create the project Trainer instance."""

    use_amp = torch.cuda.is_available() and not args.no_amp

    trainer_candidates = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "device": device,
        "use_amp": use_amp,
        "amp": use_amp,
        "checkpoint_dir": args.checkpoint_dir,
    }

    kwargs = supported_kwargs(
        Trainer,
        trainer_candidates,
    )

    print(f"Trainer arguments: {list(kwargs.keys())}")

    return Trainer(**kwargs)


def run_training(
    trainer: Trainer,
    train_loader: Any,
    validation_loader: Any,
    args: argparse.Namespace,
) -> None:
    """Run the Trainer.fit method using its supported parameter names."""

    fit_candidates = {
        "train_loader": train_loader,
        "training_loader": train_loader,
        "val_loader": validation_loader,
        "valid_loader": validation_loader,
        "validation_loader": validation_loader,
        "test_loader": validation_loader,
        "epochs": args.epochs,
        "num_epochs": args.epochs,
        "start_epoch": 0,
    }

    start_epoch = 0

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint was not found: {resume_path}"
            )

        if not hasattr(trainer, "resume"):
            raise AttributeError(
                "Trainer does not implement a resume() method."
            )

        start_epoch = trainer.resume(str(resume_path))
        fit_candidates["start_epoch"] = start_epoch

        print(
            f"Resumed from {resume_path}. "
            f"Training starts at epoch {start_epoch}."
        )

    fit_kwargs = supported_kwargs(
        trainer.fit,
        fit_candidates,
    )

    fit_signature = inspect.signature(trainer.fit)
    required_parameters = [
        name
        for name, parameter in fit_signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    missing = [
        name
        for name in required_parameters
        if name not in fit_kwargs
    ]

    if missing:
        raise RuntimeError(
            "Could not automatically match Trainer.fit() parameters. "
            f"Missing parameters: {missing}. "
            f"Actual signature: {fit_signature}"
        )

    print(f"Trainer.fit arguments: {list(fit_kwargs.keys())}")

    trainer.fit(**fit_kwargs)


def main() -> None:
    """Configure and launch MTCF-Net training."""

    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("MTCF-Net PAMAP2 Training")
    print("=" * 60)
    print(f"Device         : {device}")
    print(f"Data directory : {args.data_dir}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Learning rate  : {args.lr}")
    print(f"AMP enabled    : {torch.cuda.is_available() and not args.no_amp}")

    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")

    train_loader, validation_loader = create_loaders(args)

    print(f"Training batches   : {len(train_loader)}")
    print(f"Validation batches : {len(validation_loader)}")

    model = create_model()
    model = model.to(device)

    parameter_count = count_trainable_parameters(model)

    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.01,
    )

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
    )

    run_training(
        trainer=trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        args=args,
    )

    print("=" * 60)
    print("Training completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
