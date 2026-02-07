import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.model import Ensemble, SlidingWindowWrapper

import zipfile
import os
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    if "val" in dataloaders:
        output_part = "val"
    elif "test" in dataloaders:
        output_part = "test"
    else:
        output_part = next(iter(dataloaders.keys()))
    tta_transforms = instantiate(config.tta_transforms)
    postprocess_transforms = instantiate(config.postprocess_transforms)

    model = instantiate(config.model).to(device)
    assert isinstance(model, SlidingWindowWrapper)
    assert not (isinstance(model.model, Ensemble) and tta_transforms)

    metrics = instantiate(config.metrics)

    is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
    from_pretrained_paths = config.inferencer.from_pretrained
    if isinstance(from_pretrained_paths, str) == 1:
        run_name = Path(from_pretrained_paths).stem
    else:
        run_name = '_'.join([Path(path).stem for path in from_pretrained_paths])

    if is_kaggle_env:
        save_path = Path('/kaggle/working/')
    else:
        save_path = ROOT_PATH / "data" / "predicted" / run_name
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        tta_transforms=tta_transforms,
        postprocess_transforms=postprocess_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    if is_kaggle_env:
        root_dir = Path('/kaggle/input/vesuvius-challenge-surface-detection')
        zip_path = "/kaggle/working/submission.zip"

        test_df = pd.read_csv(f"{root_dir}/test.csv")

        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for image_id in test_df["id"]:
                out_path = save_path / output_part / f"{image_id}.tif"
                z.write(out_path, arcname=f"{image_id}.tif")

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
