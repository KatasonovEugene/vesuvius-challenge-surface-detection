import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np

import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_utils import view_binary_mask_3d


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


def _iter_tif_files(path: Path) -> Iterator[Path]:
    if path.is_dir():
        for tif_path in sorted(path.rglob("*.tif")):
            if tif_path.is_file():
                yield tif_path
        return
    if path.is_file() and path.suffix.lower() == ".tif":
        yield path


def _view_tif_array(mask: np.ndarray, label: str, index: int, total: int) -> None:
    print(f"Viewing binary mask {index}/{total}: {label} (shape: {mask.shape})")
    view_binary_mask_3d(mask)

# =======================================================================
#     THIS **** DOESNT WORK AT ALL (IDK WHY). TRY IT YOURSELF
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "View tiff files binary masks in 3d (with napari). "
        )
    )
    parser.add_argument("input_path", help="Path to masks zip, a folder, or a .tif file")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as gt_zf:
            index = _build_index(gt_zf)
            total = len(index)
            for i, image_id in enumerate(index, start=1):
                gt = _read_tif_from_zip(gt_zf, index[image_id])
                _view_tif_array(gt, image_id, i, total)
        return 0

    tif_files = list(_iter_tif_files(input_path))
    if not tif_files:
        print(f"No .tif files found at: {input_path}")
        return 1

    total = len(tif_files)
    for i, tif_path in enumerate(tif_files, start=1):
        gt = tifffile.imread(str(tif_path))
        _view_tif_array(gt, tif_path.stem, i, total)

    return 0


if __name__ == "__main__":
    sys.exit(main())
