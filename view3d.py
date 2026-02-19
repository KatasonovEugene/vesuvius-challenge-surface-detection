import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

#========= I DONT KNOW HOW TO ADD CONFIGS TO THIS (OR CLI) ==================

INPUT_PATH = 'data/train_labels'  # Change this to your desired path

#============================================================================

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

@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if len(config.writer.run_name) == 0:
        config.writer.run_name = "plug"

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    # logger = setup_saving_and_logging(config)
    # writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    # assert hasattr(model, "get_inner_model")

    loss_function = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    epoch_len = config.trainer.get("epoch_len")

    if 'steps' in config.lr_scheduler:
        config.lr_scheduler['steps'] = config.trainer.n_epochs * epoch_len
    if 'max_epochs' in config.lr_scheduler:
        config.lr_scheduler['max_epochs'] = config.trainer.n_epochs

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)


    # input_path = Path(config.input_path)
    input_path = Path(INPUT_PATH)

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

if __name__ == "__main__":
    main()
