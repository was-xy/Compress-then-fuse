"""Run helpers: loggers, seeding, the fold protocol and per-fold summaries."""

import logging
import random
from typing import List, Sequence

import numpy as np
import torch

#: Fold -> (training folds, validation fold, test fold), the protocol of the
#: PASTIS benchmark.
FOLD_SEQUENCE = {
    1: ([1, 2, 3], [4], [5]),
    2: ([2, 3, 4], [5], [1]),
    3: ([3, 4, 5], [1], [2]),
    4: ([4, 5, 1], [2], [3]),
    5: ([5, 1, 2], [3], [4]),
}


def get_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(console)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_folds(spec: str) -> List[int]:
    if spec.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds = [int(value) for value in spec.split(",")]
    for fold in folds:
        assert fold in FOLD_SEQUENCE, f"fold must be in 1-5, got {fold}"
    return folds


def summarise_folds(metrics: Sequence[dict]) -> dict:
    """Mean and standard deviation of the per-fold test metrics."""
    mious = [m["mIoU"] for m in metrics]
    accuracies = [m["Accuracy"] for m in metrics]
    return {
        "n_folds": len(metrics),
        "folds": [m["fold"] for m in metrics],
        "mIoU_mean": float(np.mean(mious)),
        "mIoU_std": float(np.std(mious)),
        "Accuracy_mean": float(np.mean(accuracies)),
        "Accuracy_std": float(np.std(accuracies)),
        "per_fold_mIoU": {str(m["fold"]): m["mIoU"] for m in metrics},
        "per_fold_Accuracy": {str(m["fold"]): m["Accuracy"] for m in metrics},
    }
