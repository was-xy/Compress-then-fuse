"""PASTIS-R loader.

The dataset returns, for one patch,

    ((images, dates), target)

where ``images[modality]`` has shape (T, C, H, W) and ``dates[modality]`` holds
the acquisition days counted from the reference date.  Modalities keep their
own temporal grid unless the experiment asks for early fusion, in which case
every modality is resampled onto the S2 grid.

Expected directory layout::

    <root>/metadata.geojson
    <root>/NORM_<source>_patch.json
    <root>/DATA_<source>/<source>_<patch_id>.npy
    <root>/ANNOTATIONS/TARGET_<patch_id>.npy
"""

import collections.abc
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from scipy.interpolate import interp1d
from torch.nn import functional as F

from configs.experiments import HYBRID_SOURCES

# Void label of PASTIS; remapped to the loss ignore index.
VOID_LABEL = 19
IGNORE_INDEX = -100


class PASTIS(tdata.Dataset):
    def __init__(
        self,
        root: str,
        sources: Sequence[str],
        folds: Sequence[int],
        hybrid: bool = False,
        align_to_s2: bool = False,
        reference_date: str = "2018-09-01",
        drop_rate_s2: float = 0.0,
        drop_rate_s1: float = 0.0,
        early_season_days: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.sources = list(sources)
        self.hybrid = hybrid
        self.align_to_s2 = align_to_s2
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.drop_rate_s2 = drop_rate_s2
        self.drop_rate_s1 = drop_rate_s1
        self.early_season_days = early_season_days

        metadata = gpd.read_file(os.path.join(root, "metadata.geojson"))
        metadata.index = metadata["ID_PATCH"].astype(int)
        metadata.sort_index(inplace=True)

        self.acquisition_days = {
            source: self._read_acquisition_days(metadata, source)
            for source in self.sources
        }

        metadata = pd.concat([metadata[metadata["Fold"] == f] for f in folds])
        self.patch_ids = list(metadata.index)

        self.norm = {source: self._read_norm(source, folds) for source in self.sources}

    # ---- metadata ----

    def _read_acquisition_days(self, metadata, source) -> Dict[int, np.ndarray]:
        """Acquisition day of every frame, in the order the frames are stored."""
        days = {}
        for patch_id, sequence in metadata[f"dates-{source}"].items():
            if isinstance(sequence, (str, bytes)):
                sequence = json.loads(sequence)
            stamps = pd.DataFrame().from_dict(sequence, orient="index")[0]
            offsets = stamps.apply(
                lambda x: (
                    datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                    - self.reference_date
                ).days
            )
            days[patch_id] = offsets.values.astype(np.int64)
        return days

    def _read_norm(self, source, folds):
        with open(os.path.join(self.root, f"NORM_{source}_patch.json")) as handle:
            values = json.load(handle)
        mean = np.stack([values[f"Fold_{f}"]["mean"] for f in folds]).mean(axis=0)
        std = np.stack([values[f"Fold_{f}"]["std"] for f in folds]).mean(axis=0)
        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

    # ---- temporal operators ----

    def _drop_rate(self, key: str) -> float:
        return self.drop_rate_s1 if "S1" in key else self.drop_rate_s2

    @staticmethod
    def _keep_mask(length: int, drop_rate: float) -> np.ndarray:
        if drop_rate <= 0:
            return np.ones(length, dtype=bool)
        n_keep = max(2, int(length * (1 - drop_rate)))
        mask = np.zeros(length, dtype=bool)
        mask[np.sort(np.random.choice(length, size=n_keep, replace=False))] = True
        return mask

    @staticmethod
    def _resample(images, dates, target_dates):
        if torch.equal(dates, target_dates):
            return images, target_dates.clone()
        interpolator = interp1d(
            dates.numpy(), images.numpy(), axis=0, fill_value="extrapolate"
        )
        resampled = torch.from_numpy(interpolator(target_dates.numpy())).float()
        return resampled, target_dates.clone()

    def _truncate_season(self, images, dates):
        if self.early_season_days is None:
            return images, dates
        for key in list(images):
            mask = dates[key] <= self.early_season_days
            if not mask.any():
                mask[dates[key].abs().argmin()] = True
            images[key] = images[key][mask]
            dates[key] = dates[key][mask]
        return images, dates

    def _merge_coherence(self, images, dates):
        """Resample each coherence source onto its SAR grid and concatenate."""
        merged, merged_dates = {}, {}
        consumed = {s for pair in HYBRID_SOURCES.values() for s in pair}
        for key in images:
            if key not in consumed:
                merged[key], merged_dates[key] = images[key], dates[key]

        for modality, (sar, coherence) in HYBRID_SOURCES.items():
            resampled, _ = self._resample(
                images[coherence], dates[coherence], dates[sar]
            )
            merged[modality] = torch.cat([images[sar], resampled], dim=1)
            merged_dates[modality] = dates[sar].clone()
        return merged, merged_dates

    def _align_to_s2(self, images, dates):
        for key in list(images):
            if key != "S2":
                images[key], dates[key] = self._resample(
                    images[key], dates[key], dates["S2"]
                )
        return images, dates

    def _drop(self, images, dates, shared_key: Optional[str] = None):
        if self.drop_rate_s1 <= 0 and self.drop_rate_s2 <= 0:
            return images, dates

        if shared_key is not None:
            shared = self._keep_mask(
                images[shared_key].shape[0], self._drop_rate(shared_key)
            )
            masks = {key: shared for key in images}
        else:
            masks = {
                key: self._keep_mask(images[key].shape[0], self._drop_rate(key))
                for key in images
            }

        return (
            {key: images[key][masks[key]] for key in images},
            {key: dates[key][masks[key]] for key in dates},
        )

    # ---- sampling ----

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]

        images, dates = {}, {}
        for source in self.sources:
            array = np.load(
                os.path.join(self.root, f"DATA_{source}", f"{source}_{patch_id}.npy")
            ).astype(np.float32)
            mean, std = self.norm[source]
            images[source] = (torch.from_numpy(array) - mean[None, :, None, None]) / std[
                None, :, None, None
            ]
            dates[source] = torch.from_numpy(self.acquisition_days[source][patch_id])

        target = np.load(
            os.path.join(self.root, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        )
        target = torch.from_numpy(target[0].astype(int))
        target[target == VOID_LABEL] = IGNORE_INDEX

        images, dates = self._truncate_season(images, dates)
        if self.hybrid:
            images, dates = self._merge_coherence(images, dates)
        if self.align_to_s2:
            images, dates = self._align_to_s2(images, dates)
            images, dates = self._drop(images, dates, shared_key="S2")
        else:
            images, dates = self._drop(images, dates)

        return (images, dates), target


def pad_tensor(x: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
    padding = [0 for _ in range(2 * (x.dim() - 1))] + [0, length - x.shape[0]]
    return F.pad(x, pad=padding, value=pad_value)


def pad_collate(batch, pad_value: float = 0.0):
    """Collate that right-pads the temporal axis to the longest sequence."""
    element = batch[0]

    if isinstance(element, torch.Tensor):
        lengths = [item.shape[0] for item in batch]
        if element.dim() > 0 and len(set(lengths)) > 1:
            batch = [pad_tensor(item, max(lengths), pad_value) for item in batch]
        return torch.stack(batch, dim=0)

    if isinstance(element, collections.abc.Mapping):
        return {
            key: pad_collate([sample[key] for sample in batch], pad_value)
            for key in element
        }

    if isinstance(element, collections.abc.Sequence):
        return [pad_collate(samples, pad_value) for samples in zip(*batch)]

    raise TypeError(f"unsupported batch element: {type(element)}")


def build_datasets(cfg, root: str, train_folds, val_fold, test_fold) -> List[PASTIS]:
    """Train / validation / test splits for one experiment and one fold.

    Temporal dropout is a training-time augmentation; it is also applied to the
    validation split so that model selection sees the same input regime.  The
    test split always keeps every acquisition.
    """
    common = dict(
        root=root,
        sources=cfg.sources,
        hybrid=cfg.is_hybrid,
        align_to_s2=cfg.align_to_s2,
        early_season_days=cfg.early_season_days,
    )
    return [
        PASTIS(
            **common,
            folds=train_folds,
            drop_rate_s2=cfg.drop_rate_s2,
            drop_rate_s1=cfg.drop_rate_s1,
        ),
        PASTIS(
            **common,
            folds=val_fold,
            drop_rate_s2=cfg.drop_rate_s2,
            drop_rate_s1=cfg.drop_rate_s1,
        ),
        PASTIS(**common, folds=test_fold),
    ]
