import matplotlib
# matplotlib.use('Agg')

import napari
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from src.utils.io_utils import ROOT_PATH

def view3d(volume, gt_mask, gt_skel, outputs=None, sample_idx=0, **batch):
    volume = volume.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    gt_skel = gt_skel.cpu().numpy()
    if outputs is not None:
        outputs = outputs.cpu().numpy()

    viewer = napari.Viewer()
    viewer.add_image(gt_skel[sample_idx], name='GT Skeleton')
    viewer.add_image(gt_mask[sample_idx], name='GT Mask')
    viewer.add_image(volume[sample_idx], name='Volume')
    if outputs is not None:
        viewer.add_image(outputs[sample_idx], name='Outputs')
    viewer.dims.ndisplay = 3
    napari.run()

def view_batch_3d(volume, gt_mask, gt_skel, outputs=None, **batch):
    for i in range(volume.shape[0]):
        view3d(volume, gt_mask, gt_skel, outputs=outputs, sample_idx=i)


def plot_sample(volume, gt_mask, gt_skel, outputs=None, sample_idx=0, max_slices=16, name="sample_plot", **batch):
    volume = volume.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()
    gt_skel = gt_skel.detach().cpu().numpy()
    if outputs is not None:
        outputs = outputs.detach().cpu().numpy()

    img = np.squeeze(volume[sample_idx])  # (D, H, W)
    mask = np.squeeze(gt_mask[sample_idx])  # (D, H, W)
    skel_mask = np.squeeze(gt_skel[sample_idx])  # (D, H, W)
    output_mask = None
    if outputs is not None:
        output_mask = np.squeeze(outputs[sample_idx])  # (D, H, W)
    D = img.shape[0]

    # Decide which slices to plot
    step = max(1, D // max_slices)
    slices = range(0, D, step)

    n_slices = len(slices)
    
    if output_mask is not None:
        fig, axes = plt.subplots(4, n_slices, figsize=(4*n_slices, 8))
    else:
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

        if output_mask is not None:
            axes[3, i].imshow(output_mask[s], cmap='gray')
            axes[3, i].set_title(f"Output Mask {s}")
            axes[3, i].axis('off')

    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout()

    is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
    if is_kaggle_env:
        plot_save_path = Path('plots')
    else:
        plot_save_path = ROOT_PATH / 'plots' / 'last_run'
    plot_save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_save_path / f'{name}_{sample_idx}.png')
    plt.close()

def plot_results(gt_mask, gt_skel, outputs_probs, outputs_post_processed, sample_idx=0, max_slices=16, prefix="", name="result_plot", **batch):
    gt_mask = gt_mask.detach().cpu().numpy()
    gt_skel = gt_skel.detach().cpu().numpy()
    outputs_probs = outputs_probs.detach().cpu().numpy()
    outputs_post_processed = outputs_post_processed.detach().cpu().numpy()

    gt_mask = np.squeeze(gt_mask[sample_idx])  # (D, H, W)
    gt_skel = np.squeeze(gt_skel[sample_idx])  # (D, H, W)
    outputs_probs = np.squeeze(outputs_probs[sample_idx])  # (D, H, W)
    outputs_post_processed = np.squeeze(outputs_post_processed[sample_idx])  # (D, H, W)

    D = gt_mask.shape[0]
    # Decide which slices to plot
    step = max(1, D // max_slices)
    slices = range(0, D, step)

    n_slices = len(slices)
    
    fig, axes = plt.subplots(4, n_slices, figsize=(4*n_slices, 8))

    for i, s in enumerate(slices):
        axes[0, i].imshow(gt_mask[s], cmap='gray')
        axes[0, i].set_title(f"Mask {s}")
        axes[0, i].axis('off')

        axes[1, i].imshow(gt_skel[s], cmap='gray')
        axes[1, i].set_title(f"Skeleton Mask {s}")
        axes[1, i].axis('off')

        axes[2, i].imshow(outputs_probs[s], cmap='gray')
        axes[2, i].set_title(f"{prefix}Output Probabilities {s}")
        axes[2, i].axis('off')

        axes[3, i].imshow(outputs_post_processed[s], cmap='gray')
        axes[3, i].set_title(f"{prefix}Output Post-Processed {s}")
        axes[3, i].axis('off')

    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout()

    is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
    if is_kaggle_env:
        plot_save_path = Path('plots')
    else:
        plot_save_path = ROOT_PATH / 'plots' / 'last_run'
    plot_save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_save_path / f'{prefix}_{name}_{sample_idx}.png')
    plt.close()


def plot_batch(volume, gt_mask, gt_skel, outputs=None, max_slices=16, name="sample_plot", **batch):
    for i in range(volume.shape[0]):
        plot_sample(volume, gt_mask, gt_skel, outputs=outputs, sample_idx=i, max_slices=max_slices, name=name)
