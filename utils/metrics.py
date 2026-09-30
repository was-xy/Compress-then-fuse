"""Confusion matrix and the metrics derived from it."""

from typing import Dict, Tuple

import numpy as np
import torch


class ConfusionMatrix:
    """Running confusion matrix, accumulated on the device of the predictions."""

    def __init__(self, num_classes: int, ignore_index=None, device: str = "cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        predicted = predicted.reshape(-1).to(self.device)
        target = target.reshape(-1).to(self.device)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            predicted, target = predicted[keep], target[keep]
        index = target * self.num_classes + predicted
        self.matrix += torch.bincount(
            index, minlength=self.num_classes ** 2
        ).view(self.num_classes, self.num_classes)

    def value(self) -> np.ndarray:
        return self.matrix.cpu().numpy()

    def miou_accuracy(self) -> Tuple[float, float]:
        """Mean intersection over union and overall accuracy, in percent."""
        matrix = self.value()
        intersection = np.diag(matrix)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = intersection / union
        total = matrix.sum()
        accuracy = intersection.sum() / total if total else 0.0
        return float(np.nanmean(iou) * 100), float(accuracy * 100)


def analyse_confusion_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Per-class IoU plus the macro averaged scores, all in percent.

    Classes that never appear, either as a label or as a prediction, are left
    out of the macro average.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    intersection = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersection / (predicted + actual - intersection)
        precision = intersection / predicted
        recall = intersection / actual
        f1 = 2 * intersection / (predicted + actual)

    metrics = {
        "mIoU": float(np.nanmean(iou) * 100),
        "Precision": float(np.nanmean(precision) * 100),
        "Recall": float(np.nanmean(recall) * 100),
        "F1": float(np.nanmean(f1) * 100),
        "Accuracy": float(intersection.sum() / matrix.sum() * 100),
    }
    metrics["per_class_IoU"] = {
        str(i): (float(v * 100) if np.isfinite(v) else None)
        for i, v in enumerate(iou)
    }
    return metrics
