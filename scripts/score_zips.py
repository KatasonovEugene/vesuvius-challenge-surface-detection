import argparse
import io
import sys
import zipfile
from typing import Dict, Iterable, Tuple

import numpy as np

import tifffile

from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics import (
    compute_leaderboard_score,
)


def _read_tif_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as fp:
        data = fp.read()
    return tifffile.imread(io.BytesIO(data))


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
            "Compute leaderboard score for matching .tif files in two zip archives. "
            "The first zip is predictions, the second is ground truth."
        )
    )
    parser.add_argument("pred_zip", help="Path to predictions zip")
    parser.add_argument("gt_zip", help="Path to ground-truth zip")
    args = parser.parse_args()

    with zipfile.ZipFile(args.pred_zip, "r") as pred_zf, zipfile.ZipFile(
        args.gt_zip, "r"
    ) as gt_zf:
        pred_index = _build_index(pred_zf)
        gt_index = _build_index(gt_zf)

        common_ids = sorted(set(pred_index) & set(gt_index))
        if not common_ids:
            print("No matching .tif files found by {image_id}.tif name.")
            return 1

        header = (
            "image_id\tleaderboard_score\ttopo_score\tsurface_dice"
            "\tvoi_score\tvoi_split\tvoi_merge"
            "\tavg_leaderboard_score\tavg_topo_score\tavg_surface_dice"
            "\tavg_voi_score\tavg_voi_split\tavg_voi_merge"
        )
        print(header)

        totals = np.zeros(6, dtype=np.float64)
        count = 0

        for i, image_id in enumerate(common_ids):
            pr = _read_tif_from_zip(pred_zf, pred_index[image_id])
            gt = _read_tif_from_zip(gt_zf, gt_index[image_id])

            scores = _score_pair(pr.astype(np.uint8), gt.astype(np.uint8))
            totals += np.asarray(scores, dtype=np.float64)
            count += 1
            averages = totals / count
            line = (
                f"{image_id} ({i + 1}/{len(common_ids)})\t{scores[0]:.6f}\t{scores[1]:.6f}\t{scores[2]:.6f}"
                f"\t{scores[3]:.6f}\t{scores[4]:.6f}\t{scores[5]:.6f}"
                f"\t{averages[0]:.6f}\t{averages[1]:.6f}\t{averages[2]:.6f}"
                f"\t{averages[3]:.6f}\t{averages[4]:.6f}\t{averages[5]:.6f}"
            )
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
