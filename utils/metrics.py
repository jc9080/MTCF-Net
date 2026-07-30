from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Classification metrics computed from model predictions.

    Attributes:
        accuracy:
            Overall classification accuracy.
        macro_precision:
            Mean precision across all classes.
        macro_recall:
            Mean recall across all classes.
        macro_f1:
            Mean F1 score across all classes.
        confusion_matrix:
            Confusion matrix with shape (num_classes, num_classes).
            Rows represent ground-truth classes and columns represent
            predicted classes.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: Tensor

    def to_dict(self) -> Dict[str, float]:
        """
        Convert scalar metrics to a dictionary.

        The confusion matrix is intentionally excluded because it is a tensor.
        """
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def build_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """
    Build a multiclass confusion matrix.

    Args:
        predictions:
            Predicted class indices with shape (N,).
        targets:
            Ground-truth class indices with shape (N,).
        num_classes:
            Number of classes.

    Returns:
        Confusion matrix with shape (num_classes, num_classes).

    Raises:
        ValueError:
            If tensor shapes, class counts, or label ranges are invalid.
    """
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")

    predictions = predictions.detach().reshape(-1).to(torch.long)
    targets = targets.detach().reshape(-1).to(torch.long)

    if predictions.numel() != targets.numel():
        raise ValueError(
            "predictions and targets must contain the same number of elements, "
            f"but received {predictions.numel()} and {targets.numel()}."
        )

    if predictions.numel() == 0:
        raise ValueError("predictions and targets must not be empty.")

    if predictions.min().item() < 0 or predictions.max().item() >= num_classes:
        raise ValueError(
            "predictions contain a class index outside the valid range "
            f"[0, {num_classes - 1}]."
        )

    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError(
            "targets contain a class index outside the valid range "
            f"[0, {num_classes - 1}]."
        )

    encoded = targets * num_classes + predictions

    confusion_matrix = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )

    return confusion_matrix.reshape(num_classes, num_classes)


def compute_metrics_from_confusion_matrix(
    confusion_matrix: Tensor,
    epsilon: float = 1e-12,
) -> ClassificationMetrics:
    """
    Compute classification metrics from a confusion matrix.

    Args:
        confusion_matrix:
            Square confusion matrix. Rows are ground truth and columns are
            predictions.
        epsilon:
            Numerical stability constant.

    Returns:
        ClassificationMetrics containing accuracy and macro-averaged scores.
    """
    if confusion_matrix.ndim != 2:
        raise ValueError(
            "confusion_matrix must be a 2D tensor, "
            f"but received shape {tuple(confusion_matrix.shape)}."
        )

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be square, "
            f"but received shape {tuple(confusion_matrix.shape)}."
        )

    if confusion_matrix.numel() == 0:
        raise ValueError("confusion_matrix must not be empty.")

    matrix = confusion_matrix.to(torch.float64)

    true_positive = torch.diag(matrix)
    predicted_count = matrix.sum(dim=0)
    target_count = matrix.sum(dim=1)
    total_count = matrix.sum()

    if total_count.item() == 0:
        raise ValueError("confusion_matrix contains no samples.")

    precision = true_positive / predicted_count.clamp_min(epsilon)
    recall = true_positive / target_count.clamp_min(epsilon)

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall).clamp_min(epsilon)
    )

    accuracy = true_positive.sum() / total_count

    # Macro metrics are averaged only over classes present
    # in the ground-truth targets.
    present_classes = target_count > 0

    macro_precision = precision[present_classes].mean()
    macro_recall = recall[present_classes].mean()
    macro_f1 = f1[present_classes].mean()

    return ClassificationMetrics(
        accuracy=float(accuracy.item()),
        macro_precision=float(macro_precision.item()),
        macro_recall=float(macro_recall.item()),
        macro_f1=float(macro_f1.item()),
        confusion_matrix=confusion_matrix.detach().clone(),
    )


def compute_classification_metrics(
    logits_or_predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    """
    Compute multiclass accuracy, macro precision, recall, and F1.

    Args:
        logits_or_predictions:
            Either logits with shape (N, num_classes) or predicted class
            indices with shape (N,).
        targets:
            Ground-truth class indices with shape (N,).
        num_classes:
            Number of classes.

    Returns:
        ClassificationMetrics.

    Raises:
        ValueError:
            If input dimensions are unsupported or inconsistent.
    """
    if logits_or_predictions.ndim == 2:
        if logits_or_predictions.shape[1] != num_classes:
            raise ValueError(
                f"Expected logits with {num_classes} classes, "
                f"but received shape {tuple(logits_or_predictions.shape)}."
            )

        predictions = logits_or_predictions.argmax(dim=1)

    elif logits_or_predictions.ndim == 1:
        predictions = logits_or_predictions

    else:
        raise ValueError(
            "logits_or_predictions must have shape (N, C) or (N,), "
            f"but received {tuple(logits_or_predictions.shape)}."
        )

    confusion_matrix = build_confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
    )

    return compute_metrics_from_confusion_matrix(confusion_matrix)


class MetricAccumulator:
    """
    Accumulate predictions across mini-batches without storing logits.

    This class stores only a confusion matrix, so its memory usage is constant
    with respect to the number of samples.
    """

    def __init__(self, num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.num_classes = num_classes
        self._confusion_matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        )

    @torch.no_grad()
    def update(
        self,
        logits_or_predictions: Tensor,
        targets: Tensor,
    ) -> None:
        """
        Add one mini-batch to the accumulated confusion matrix.
        """
        if logits_or_predictions.ndim == 2:
            if logits_or_predictions.shape[1] != self.num_classes:
                raise ValueError(
                    f"Expected logits with {self.num_classes} classes, "
                    f"but received shape {tuple(logits_or_predictions.shape)}."
                )

            predictions = logits_or_predictions.argmax(dim=1)

        elif logits_or_predictions.ndim == 1:
            predictions = logits_or_predictions

        else:
            raise ValueError(
                "logits_or_predictions must have shape (N, C) or (N,), "
                f"but received {tuple(logits_or_predictions.shape)}."
            )

        batch_matrix = build_confusion_matrix(
            predictions=predictions.cpu(),
            targets=targets.cpu(),
            num_classes=self.num_classes,
        )

        self._confusion_matrix += batch_matrix

    def compute(self) -> ClassificationMetrics:
        """
        Compute metrics from all accumulated mini-batches.
        """
        return compute_metrics_from_confusion_matrix(
            self._confusion_matrix
        )

    def reset(self) -> None:
        """
        Clear all accumulated values.
        """
        self._confusion_matrix.zero_()

    @property
    def confusion_matrix(self) -> Tensor:
        """
        Return a copy of the accumulated confusion matrix.
        """
        return self._confusion_matrix.clone()


def _run_self_test() -> None:
    """
    Run deterministic correctness checks.
    """
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 2.0],
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ]
    )

    targets = torch.tensor([0, 1, 2, 2, 0, 1])

    metrics = compute_classification_metrics(
        logits_or_predictions=logits,
        targets=targets,
        num_classes=3,
    )

    expected_matrix = torch.tensor(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 1, 1],
        ]
    )

    if not torch.equal(metrics.confusion_matrix, expected_matrix):
        raise RuntimeError(
            "Confusion matrix self-test failed.\n"
            f"Expected:\n{expected_matrix}\n"
            f"Received:\n{metrics.confusion_matrix}"
        )

    expected_accuracy = 5.0 / 6.0

    if abs(metrics.accuracy - expected_accuracy) > 1e-8:
        raise RuntimeError(
            "Accuracy self-test failed: "
            f"expected {expected_accuracy}, received {metrics.accuracy}."
        )

    accumulator = MetricAccumulator(num_classes=3)
    accumulator.update(logits[:3], targets[:3])
    accumulator.update(logits[3:], targets[3:])

    accumulated_metrics = accumulator.compute()

    if not torch.equal(
        accumulated_metrics.confusion_matrix,
        expected_matrix,
    ):
        raise RuntimeError("MetricAccumulator self-test failed.")

    print("Confusion matrix:")
    print(metrics.confusion_matrix)
    print()
    print(f"Accuracy:        {metrics.accuracy:.6f}")
    print(f"Macro precision:{metrics.macro_precision:.6f}")
    print(f"Macro recall:   {metrics.macro_recall:.6f}")
    print(f"Macro F1:       {metrics.macro_f1:.6f}")
    print()
    print("All metric tests passed.")


if __name__ == "__main__":
    _run_self_test()
