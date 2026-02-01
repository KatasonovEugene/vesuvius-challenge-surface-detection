import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from src.utils.io_utils import ROOT_PATH

def plot_sample(volume, gt_mask, gt_skel, sample_idx=0, max_slices=16, **batch):
    img = np.squeeze(volume[sample_idx])  # (D, H, W)
    mask = np.squeeze(gt_mask[sample_idx])  # (D, H, W)
    skel_mask = np.squeeze(gt_skel[sample_idx])  # (D, H, W)
    D = img.shape[0]

    # Decide which slices to plot
    step = max(1, D // max_slices)
    slices = range(0, D, step)

    n_slices = len(slices)
    fig, axes = plt.subplots(3, n_slices, figsize=(3*n_slices, 6))

    for i, s in enumerate(slices):
        axes[0, i].imshow(img[s], cmap='gray')
        axes[0, i].set_title(f"Slice {s}")
        axes[0, i].axis('off')

        axes[1, i].imshow(mask[s], cmap='gray')
        axes[1, i].set_title(f"Mask {s}")
        axes[1, i].axis('off')

        axes[2, i].imshow(skel_mask[s], cmap='gray')
        axes[2, i].set_title(f"Skeleton Mask {s}")
        axes[2, i].axis('off')

    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout()

    is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
    if is_kaggle_env:
        plot_save_path = Path('plots')
    else:
        plot_save_path = ROOT_PATH / 'plots'
    plot_save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_save_path / f'sample_plot_{sample_idx}.png')


def plot_batch(volume, gt_mask, gt_skel, max_slices=16, **batch):
    for i in range(volume.shape[0]):
        plot_sample(volume, gt_mask, gt_skel, sample_idx=i, max_slices=max_slices)
