"""Write a tiny synthetic PASTIS-R tree, for smoke testing without the dataset.

The patches are random noise, so the numbers are meaningless; the point is that
the loader, the collate function, every fusion strategy and the training loop
can be exercised end to end on a laptop.

    python -m tools.make_synthetic_data --out ./synthetic --patches 20
    python train.py --exp_id 49 --folds 1 --data_root ./synthetic \\
        --epochs 1 --batch_size 2 --num_workers 0
"""

import argparse
import json
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from configs.experiments import INPUT_DIM

#: Source -> (number of acquisitions, first day, revisit period in days).
SOURCES = {
    "S2": (12, 4, 25),
    "S1A": (10, 6, 30),
    "S1D": (9, 9, 33),
    "S1ABcoh": (8, 7, 36),
    "S1DBcoh": (8, 11, 36),
}
NUM_CLASSES = 19


def acquisition_dates(n_dates, first_day, period):
    """Dates as YYYYMMDD integers, starting from the reference date."""
    days = np.arange(n_dates) * period + first_day
    origin = np.datetime64("2018-09-01")
    return {
        str(i): int(str(origin + np.timedelta64(int(d), "D")).replace("-", ""))
        for i, d in enumerate(days)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./synthetic", type=str)
    parser.add_argument("--patches", default=20, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(os.path.join(root, "ANNOTATIONS"), exist_ok=True)
    for source in SOURCES:
        os.makedirs(os.path.join(root, f"DATA_{source}"), exist_ok=True)

    records = []
    for index in range(args.patches):
        patch_id = 10000 + index
        for source, (n_dates, first_day, period) in SOURCES.items():
            channels = INPUT_DIM[source]
            patch = rng.normal(
                loc=1000.0, scale=300.0, size=(n_dates, channels, args.size, args.size)
            )
            np.save(
                os.path.join(root, f"DATA_{source}", f"{source}_{patch_id}.npy"),
                patch.astype(np.float32),
            )
        target = rng.integers(0, NUM_CLASSES + 1, size=(3, args.size, args.size))
        np.save(
            os.path.join(root, "ANNOTATIONS", f"TARGET_{patch_id}.npy"),
            target.astype(np.int64),
        )
        record = {
            "ID_PATCH": patch_id,
            "Fold": index % 5 + 1,
            "geometry": box(index, 0, index + 1, 1),
        }
        for source, (n_dates, first_day, period) in SOURCES.items():
            record[f"dates-{source}"] = json.dumps(
                acquisition_dates(n_dates, first_day, period)
            )
        records.append(record)

    metadata = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:2154")
    metadata.to_file(os.path.join(root, "metadata.geojson"), driver="GeoJSON")

    for source in SOURCES:
        channels = INPUT_DIM[source]
        stats = {
            f"Fold_{fold}": {
                "mean": [1000.0] * channels,
                "std": [300.0] * channels,
            }
            for fold in range(1, 6)
        }
        with open(os.path.join(root, f"NORM_{source}_patch.json"), "w") as handle:
            json.dump(stats, handle)

    print(f"wrote {args.patches} synthetic patches of {args.size}x{args.size} to {root}")


if __name__ == "__main__":
    main()
