from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from utils.metrics import MetricAccumulator
from utils.checkpoint import (
    save_checkpoint,
    save_best_model,
)


class Trainer:
    """
    Generic trainer for MTCF-Net.

    Features
    --------
    - Automatic Mixed Precision (AMP)
    - Gradient Scaling
    - Accuracy / Macro F1
    - Checkpoint Saving
    - Best Model Saving
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.use_amp = (
            use_amp and torch.cuda.is_available()
        )

        self.scaler = GradScaler(
            enabled=self.use_amp
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_f1 = 0.0


    def train_one_epoch(
        self,
        train_loader,
        epoch: int,
    ):
        """
        Train one epoch.

        Returns
        -------
        loss, accuracy, macro_f1
        """

        self.model.train()

        running_loss = 0.0

        metric = MetricAccumulator(num_classes=self.model.num_classes)

        progress = tqdm(
            train_loader,
            desc=f"Train [{epoch}]",
            leave=False,
        )

        for batch in progress:

            inputs, targets = batch

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            with autocast(enabled=self.use_amp):

                logits = self.model(inputs)

                loss = self.criterion(
                    logits,
                    targets,
                )

            self.scaler.scale(loss).backward()

            self.scaler.step(
                self.optimizer,
            )

            self.scaler.update()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

            metric.update(
                logits.detach(),
                targets.detach(),
            )

            results = metric.compute()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{results.accuracy*100:.2f}%",
                f1=f"{results.macro_f1:.4f}",
            )

        epoch_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        results = metric.compute()

        return (
            epoch_loss,
            results.accuracy,
            results.macro_f1,
        )


    @torch.no_grad()
    def validate(
        self,
        val_loader,
        epoch: int,
    ):
        """
        Evaluate one epoch.

        Returns
        -------
        loss, accuracy, macro_f1
        """

        self.model.eval()

        running_loss = 0.0

        metric = MetricAccumulator(num_classes=self.model.num_classes)

        progress = tqdm(
            val_loader,
            desc=f"Valid [{epoch}]",
            leave=False,
        )

        for batch in progress:

            inputs, targets = batch

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            with autocast(enabled=self.use_amp):

                logits = self.model(inputs)

                loss = self.criterion(
                    logits,
                    targets,
                )

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

            metric.update(
                logits,
                targets,
            )

            results = metric.compute()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{results.accuracy*100:.2f}%",
                f1=f"{results.macro_f1:.4f}",
            )

        epoch_loss = (
            running_loss
            / len(val_loader.dataset)
        )

        results = metric.compute()

        return (
            epoch_loss,
            results.accuracy,
            results.macro_f1,
        )


    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        start_epoch: int = 0,
    ) -> None:
        """
        Train the model.

        Args
        ----
        train_loader
        val_loader
        epochs
        start_epoch
        """

        for epoch in range(start_epoch, epochs):

            train_loss, train_acc, train_f1 = (
                self.train_one_epoch(
                    train_loader,
                    epoch,
                )
            )

            val_loss, val_acc, val_f1 = (
                self.validate(
                    val_loader,
                    epoch,
                )
            )

            if self.scheduler is not None:
                self.scheduler.step()

            print()

            print("=" * 60)

            print(
                f"Epoch {epoch + 1}/{epochs}"
            )

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Train Acc  : {train_acc*100:.2f}%"
            )

            print(
                f"Train F1   : {train_f1:.4f}"
            )

            print()

            print(
                f"Valid Loss : {val_loss:.4f}"
            )

            print(
                f"Valid Acc  : {val_acc*100:.2f}%"
            )

            print(
                f"Valid F1   : {val_f1:.4f}"
            )

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            print(
                f"Learning Rate : {current_lr:.6e}"
            )

            print("=" * 60)

            self.best_f1 = save_best_model(
                model=self.model,
                save_dir=self.checkpoint_dir,
                metric=val_f1,
                best_metric=self.best_f1,
            )

            save_checkpoint(
                path=self.checkpoint_dir
                / "last_checkpoint.pth",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric=self.best_f1,
            )


    def resume(
        self,
        checkpoint_path: str,
    ) -> int:
        """
        Resume training from a checkpoint.

        Returns
        -------
        Next epoch to train.
        """

        from utils.checkpoint import load_checkpoint

        epoch, best_metric = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )

        self.best_f1 = best_metric

        print(
            f"Resumed from epoch {epoch} "
            f"(best F1={best_metric:.4f})"
        )

        return epoch + 1


    def load_best_model(
        self,
    ) -> None:
        """
        Load the best model weights.
        """

        from utils.checkpoint import load_best_model

        load_best_model(
            self.model,
            self.checkpoint_dir,
            map_location=self.device,
        )
