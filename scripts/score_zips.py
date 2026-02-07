import argparse
import io
import os
import sys
import zipfile
from typing import Dict, Iterable, Tuple

import numpy as np

import tifffile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics import (
    compute_leaderboard_score,
)


def _read_tif_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as fp:
        data = fp.read()
    return tifffile.imread(io.BytesIO(data))


def _read_tif_from_path(path: str) -> np.ndarray:
    return tifffile.imread(path)


def _tif_entries(zf: zipfile.ZipFile) -> Iterable[str]:
    for info in zf.infolist():
        if info.is_dir():
            continue
        if info.filename.lower().endswith(".tif"):
            yield info.filename


def _basename_no_ext(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base.lower().endswith(".tif"):
        return base[:-4]
    return base


def _build_index(zf: zipfile.ZipFile) -> Dict[str, str]:
    index = {}
    for name in _tif_entries(zf):
        key = _basename_no_ext(name)
        index[key] = name
    return index


def _build_dir_index(root_dir: str) -> Dict[str, str]:
    index = {}
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(".tif"):
                continue
            key = _basename_no_ext(filename)
            index[key] = os.path.join(root, filename)
    return index


def _score_pair(pr: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    result = compute_leaderboard_score(
        predictions=pr,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=2.0,
        voi_connectivity=26,
        voi_transform="one_over_one_plus",
        voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    return (
        result.score,
        result.topo.toposcore,
        result.surface_dice,
        result.voi.voi_score,
        result.voi.voi_split,
        result.voi.voi_merge,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute leaderboard score for matching .tif files in a predictions zip "
            "and a ground-truth directory."
        )
    )
    parser.add_argument("pred_zip", help="Path to predictions zip")
    parser.add_argument("gt_dir", help="Path to ground-truth directory (e.g., data/train_labels)")
    args = parser.parse_args()

    with zipfile.ZipFile(args.pred_zip, "r") as pred_zf:
        pred_index = _build_index(pred_zf)
        gt_index = _build_dir_index(args.gt_dir)

        common_ids = sorted(set(pred_index) & set(gt_index))
        if not common_ids:
            print("No matching .tif files found by {image_id}.tif name.")
            return 1

        print("metrics_per_image (one metric per line)")

        totals = np.zeros(6, dtype=np.float64)
        count = 0

        for i, image_id in enumerate(common_ids):
            pr = _read_tif_from_zip(pred_zf, pred_index[image_id])
            gt = _read_tif_from_path(gt_index[image_id])

            scores = _score_pair(pr.astype(np.uint8), gt.astype(np.uint8))
            totals += np.asarray(scores, dtype=np.float64)
            count += 1
            averages = totals / count
            print(f"image_id: {image_id} ({i + 1}/{len(common_ids)})")
            print(f"leaderboard_score: {scores[0]:.6f}")
            print(f"topo_score: {scores[1]:.6f}")
            print(f"surface_dice: {scores[2]:.6f}")
            print(f"voi_score: {scores[3]:.6f}")
            print(f"voi_split: {scores[4]:.6f}")
            print(f"voi_merge: {scores[5]:.6f}")
            print(f"avg_leaderboard_score: {averages[0]:.6f}")
            print(f"avg_topo_score: {averages[1]:.6f}")
            print(f"avg_surface_dice: {averages[2]:.6f}")
            print(f"avg_voi_score: {averages[3]:.6f}")
            print(f"avg_voi_split: {averages[4]:.6f}")
            print(f"avg_voi_merge: {averages[5]:.6f}")
            print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
