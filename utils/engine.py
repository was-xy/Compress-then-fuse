"""Training and evaluation loops."""

import time
from typing import Tuple

import torch

from utils.metrics import ConfusionMatrix


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return [to_device(value, device) for value in batch]


def compute_loss(output, target, criterion, aux_weight: float) -> torch.Tensor:
    loss = criterion(output["prediction"], target)
    for auxiliary in output["aux"]:
        loss = loss + aux_weight * criterion(auxiliary, target)
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    aux_weight: float,
    num_classes: int,
    ignore_index: int,
    logger,
    display_step: int = 50,
    scaler=None,
) -> Tuple[float, float]:
    model.train()
    matrix = ConfusionMatrix(num_classes, ignore_index, device=device)
    running_loss, n_steps = 0.0, 0
    start = time.time()

    for step, ((images, dates), target) in enumerate(loader, start=1):
        images, dates = to_device(images, device), to_device(dates, device)
        target = target.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images, dates)
                loss = compute_loss(output, target, criterion, aux_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images, dates)
            loss = compute_loss(output, target, criterion, aux_weight)
            loss.backward()
            optimizer.step()

        matrix.update(output["prediction"].argmax(dim=1), target)
        running_loss += loss.item()
        n_steps += 1

        if logger is not None and step % display_step == 0:
            miou, accuracy = matrix.miou_accuracy()
            logger.info(
                f"  step {step}/{len(loader)}  loss {running_loss / n_steps:.4f}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                f"mIoU {miou:.2f}  acc {accuracy:.2f}"
            )

    miou, accuracy = matrix.miou_accuracy()
    if logger is not None:
        logger.info(
            f"  train  {time.time() - start:.0f}s  "
            f"loss {running_loss / max(n_steps, 1):.4f}  "
            f"mIoU {miou:.2f}  acc {accuracy:.2f}"
        )
    return miou, accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    ignore_index: int,
    logger=None,
    tag: str = "val",
    use_amp: bool = False,
) -> Tuple[float, float, ConfusionMatrix]:
    model.eval()
    matrix = ConfusionMatrix(num_classes, ignore_index, device=device)
    running_loss, n_steps = 0.0, 0
    start = time.time()

    for (images, dates), target in loader:
        images, dates = to_device(images, device), to_device(dates, device)
        target = target.to(device).long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(images, dates)
            loss = criterion(output["prediction"], target)

        matrix.update(output["prediction"].argmax(dim=1), target)
        running_loss += loss.item()
        n_steps += 1

    miou, accuracy = matrix.miou_accuracy()
    if logger is not None:
        logger.info(
            f"  {tag}  {time.time() - start:.0f}s  "
            f"loss {running_loss / max(n_steps, 1):.4f}  "
            f"mIoU {miou:.2f}  acc {accuracy:.2f}"
        )
    return miou, accuracy, matrix
