"""Evaluate trained checkpoints on the test fold.

The checkpoint of an experiment / fold is the best validation epoch saved by
``train.py``; this script only reruns the test pass, which is useful to refresh
metrics without retraining.

    python evaluate.py --exp_id 4 --folds all --data_root /path/to/PASTIS-R
"""

import argparse
import json
import os
import pickle
from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as tdata

from configs.experiments import get_config, parse_exp_ids
from data.pastis import IGNORE_INDEX, build_datasets, pad_collate
from models.network import build_model
from utils.engine import evaluate
from utils.metrics import analyse_confusion_matrix
from utils.runtime import (
    FOLD_SEQUENCE,
    close_logger,
    get_logger,
    parse_folds,
    set_seed,
    summarise_folds,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_id", default="1", type=str)
    parser.add_argument("--folds", default="all", type=str)
    parser.add_argument("--data_root", default="./PASTIS-R", type=str)
    parser.add_argument("--res_dir", default="./results", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def evaluate_fold(cfg, args, fold, res_dir, device, logger):
    fold_dir = os.path.join(res_dir, f"Fold_{fold}")
    checkpoint_path = os.path.join(fold_dir, "model.pth.tar")
    if not os.path.isfile(checkpoint_path):
        logger.info(f"fold {fold}: no checkpoint, skipping")
        return None

    set_seed(args.seed)
    _, _, test_set = build_datasets(cfg, args.data_root, *FOLD_SEQUENCE[fold])
    loader = tdata.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(pad_collate, pad_value=cfg.pad_value),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)
    _, _, matrix = evaluate(
        model, loader, criterion, device, cfg.num_classes, IGNORE_INDEX,
        logger=logger, tag=f"test/fold{fold}",
        use_amp=args.amp and device.type == "cuda",
    )

    confusion = matrix.value()
    with open(os.path.join(fold_dir, "confusion_matrix.pkl"), "wb") as handle:
        pickle.dump(confusion, handle)

    metrics = analyse_confusion_matrix(confusion)
    metrics.update(
        fold=fold,
        seed=args.seed,
        exp_id=cfg.exp_id,
        experiment=cfg.name,
        best_val_mIoU=checkpoint.get("val_mIoU"),
        best_epoch=checkpoint.get("epoch"),
    )
    with open(os.path.join(fold_dir, "test_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.res_dir, exist_ok=True)
    logger = get_logger("evaluate", os.path.join(args.res_dir, "evaluate.log"))

    for exp_id in parse_exp_ids(args.exp_id):
        cfg = get_config(exp_id)
        res_dir = os.path.join(args.res_dir, cfg.name)
        if not os.path.isdir(res_dir):
            logger.info(f"{cfg.name}: no results directory, skipping")
            continue

        collected = []
        for fold in parse_folds(args.folds):
            metrics = evaluate_fold(cfg, args, fold, res_dir, device, logger)
            if metrics is not None:
                collected.append(metrics)
        if collected:
            summary = summarise_folds(collected)
            with open(os.path.join(res_dir, "summary.json"), "w") as handle:
                json.dump(summary, handle, indent=2)
            logger.info(
                f"{cfg.name}: mIoU {summary['mIoU_mean']:.2f} "
                f"+/- {summary['mIoU_std']:.2f} over folds {summary['folds']}"
            )

    close_logger(logger)


if __name__ == "__main__":
    main()
