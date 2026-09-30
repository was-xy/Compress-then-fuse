"""Train one or more experiments of the registry on the PASTIS-R folds.

Examples
--------
    python train.py --list
    python train.py --exp_id 4 --folds all --data_root /path/to/PASTIS-R
    python train.py --exp_id 19-29 --folds 1 --skip_done

Model selection uses the validation fold only.  The checkpoint of the best
validation epoch is the one restored for the final test pass, which is the
single number reported for a fold.
"""

import argparse
import gc
import json
import os
import pickle
import time
import traceback
from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as tdata

from configs.experiments import get_config, parse_exp_ids, print_experiments
from data.pastis import IGNORE_INDEX, build_datasets, pad_collate
from models.network import build_model, count_parameters
from utils.engine import evaluate, train_one_epoch
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
    parser.add_argument("--exp_id", default="1", type=str,
                        help="experiment id, list or range, e.g. '4', '4,7', '19-29'")
    parser.add_argument("--folds", default="all", type=str,
                        help="'all', '1' or '1,3,5'")
    parser.add_argument("--data_root", default="./PASTIS-R", type=str)
    parser.add_argument("--res_dir", default="./results", type=str)
    parser.add_argument("--list", action="store_true",
                        help="print the experiment registry and exit")
    parser.add_argument("--skip_done", action="store_true",
                        help="skip folds that already have test_metrics.json")

    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_milestones", default="50", type=str)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--display_step", default=50, type=int)
    return parser.parse_args()


def build_loaders(cfg, args, fold):
    train_folds, val_fold, test_fold = FOLD_SEQUENCE[fold]
    datasets = build_datasets(cfg, args.data_root, train_folds, val_fold, test_fold)
    collate = partial(pad_collate, pad_value=cfg.pad_value)

    loaders = []
    for dataset, shuffle in zip(datasets, (True, False, False)):
        loaders.append(
            tdata.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                drop_last=shuffle,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate,
            )
        )
    return loaders


def train_fold(cfg, args, fold, res_dir, device):
    set_seed(args.seed)
    fold_dir = os.path.join(res_dir, f"Fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    logger = get_logger(f"{cfg.name}/Fold_{fold}", os.path.join(fold_dir, "train.log"))

    train_loader, val_loader, test_loader = build_loaders(cfg, args, fold)
    logger.info(
        f"samples: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = build_model(cfg).to(device)
    logger.info(f"trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in args.lr_milestones.split(",") if m],
        gamma=args.lr_gamma,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    checkpoint_path = os.path.join(fold_dir, "model.pth.tar")
    best_val_miou, best_epoch = -1.0, -1

    for epoch in range(args.epochs):
        logger.info(f"epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            aux_weight=cfg.aux_loss_weight,
            num_classes=cfg.num_classes,
            ignore_index=IGNORE_INDEX,
            logger=logger,
            display_step=args.display_step,
            scaler=scaler,
        )
        val_miou, _, _ = evaluate(
            model, val_loader, criterion, device, cfg.num_classes, IGNORE_INDEX,
            logger=logger, tag="val", use_amp=use_amp,
        )
        scheduler.step()

        if val_miou >= best_val_miou:
            best_val_miou, best_epoch = val_miou, epoch
            torch.save(
                {"epoch": epoch, "val_mIoU": val_miou, "state_dict": model.state_dict()},
                checkpoint_path,
            )

    logger.info(f"restoring the best validation epoch ({best_epoch + 1})")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    _, _, matrix = evaluate(
        model, test_loader, criterion, device, cfg.num_classes, IGNORE_INDEX,
        logger=logger, tag="test", use_amp=use_amp,
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
        best_val_mIoU=best_val_miou,
        best_epoch=best_epoch,
    )
    with open(os.path.join(fold_dir, "test_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info(
        f"fold {fold}: val mIoU {best_val_miou:.2f}, "
        f"test mIoU {metrics['mIoU']:.2f}, "
        f"test accuracy {metrics['Accuracy']:.2f}"
    )
    close_logger(logger)

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def collect_metrics(res_dir):
    collected = []
    for fold in sorted(FOLD_SEQUENCE):
        path = os.path.join(res_dir, f"Fold_{fold}", "test_metrics.json")
        if os.path.isfile(path):
            with open(path) as handle:
                collected.append(json.load(handle))
    return collected


def run_experiment(exp_id, args, folds, device, logger):
    cfg = get_config(exp_id)
    res_dir = os.path.join(args.res_dir, cfg.name)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "config.json"), "w") as handle:
        json.dump(vars(cfg), handle, indent=2, default=str)

    logger.info(
        f"experiment {cfg.name}: modalities={cfg.modalities} fusion={cfg.fusion} "
        f"tcm={cfg.use_tcm} window={cfg.early_season_days or 'full'}"
    )

    for fold in folds:
        if args.skip_done and os.path.isfile(
            os.path.join(res_dir, f"Fold_{fold}", "test_metrics.json")
        ):
            logger.info(f"fold {fold} already done, skipping")
            continue
        start = time.time()
        train_fold(cfg, args, fold, res_dir, device)
        logger.info(f"fold {fold} finished in {(time.time() - start) / 3600:.2f} h")

    metrics = collect_metrics(res_dir)
    if metrics:
        summary = summarise_folds(metrics)
        with open(os.path.join(res_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info(
            f"{cfg.name}: mIoU {summary['mIoU_mean']:.2f} "
            f"+/- {summary['mIoU_std']:.2f} over folds {summary['folds']}"
        )


def main():
    args = parse_args()
    if args.list:
        print_experiments()
        return

    exp_ids = parse_exp_ids(args.exp_id)
    folds = parse_folds(args.folds)
    device = torch.device(args.device)
    os.makedirs(args.res_dir, exist_ok=True)
    logger = get_logger("queue", os.path.join(args.res_dir, "queue.log"))
    logger.info(f"experiments {exp_ids} on folds {folds}")

    for position, exp_id in enumerate(exp_ids, start=1):
        logger.info(f"[{position}/{len(exp_ids)}] experiment {exp_id}")
        try:
            run_experiment(exp_id, args, folds, device, logger)
        except Exception as error:  # keep the queue alive
            logger.error(f"experiment {exp_id} failed: {type(error).__name__}: {error}")
            logger.error(traceback.format_exc())
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    close_logger(logger)


if __name__ == "__main__":
    main()
