import os
from pathlib import Path

import matplotlib
import numpy as np
# import napari

HEADLESS_ENV = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
if HEADLESS_ENV:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.io_utils import ROOT_PATH


def view_binary_mask_3d(gt_mask, **batch):
    assert gt_mask.ndim == 3, "Expected gt_mask shape (D, H, W)"

    z_gradient = np.linspace(0.15, 0.85, gt_mask.shape[0])[:, None, None]
    ignore = (gt_mask == 2)
    gt_mask[ignore] = 0
    gt_mask = gt_mask.astype(float) * z_gradient

    viewer = napari.Viewer()

    viewer.theme = 'light'

    viewer.add_image(
        gt_mask,
        name='GT Mask',
        colormap='magma',
        rendering='attenuated_mip',
        attenuation=0.5,
        interpolation3d='nearest',
        interpolation2d='nearest',
        contrast_limits=[0, 1]
    )

    viewer.dims.ndisplay = 3
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'

    napari.run()
    return

def view_binary_mask_batch_3d(gt_mask, **batch):
    for i in range(gt_mask.shape[0]):
        print(f"Viewing sample {i+1}/{gt_mask.shape[0]} in batch...")
        view_binary_mask_3d(gt_mask[i], **batch)


def view3d(volume, gt_mask, gt_skel, outputs=None, sample_idx=0, **batch):
    volume = volume[sample_idx].cpu().numpy()
    gt_mask = gt_mask[sample_idx].cpu().numpy()
    gt_skel = gt_skel[sample_idx].cpu().numpy()
    if outputs is not None:
        outputs = outputs[sample_idx].detach().cpu().numpy()

    z_gradient = np.linspace(0.15, 0.85, volume.shape[0])[:, None, None]
    gt_skel = gt_skel.astype(float) * z_gradient
    if outputs is not None:
        outputs = outputs.astype(float) * z_gradient
    ignore = (gt_mask == 2)
    gt_mask[ignore] = 0
    gt_mask = gt_mask.astype(float) * z_gradient

    viewer = napari.Viewer()
    viewer.theme = 'light'

    viewer.add_image(
        gt_skel,
        name='GT Skeleton',
        colormap='magma',
        rendering='attenuated_mip',
        attenuation=0.5,
        interpolation3d='nearest',
        interpolation2d='nearest',
        contrast_limits=[0, 1]
    )
    viewer.add_image(
        gt_mask,
        name='GT Mask',
        colormap='magma',
        rendering='attenuated_mip',
        attenuation=0.5,
        interpolation3d='nearest',
        interpolation2d='nearest',
        contrast_limits=[0, 1]
    )
    viewer.add_image(volume, name='Volume')
    if outputs is not None:
        viewer.add_image(
            outputs,
            name='Outputs',
            colormap='magma',
            rendering='attenuated_mip',
            attenuation=0.5,
            contrast_limits=[0, 1]
        )

    # verts, faces, normals, values = measure.marching_cubes(gt_skel, level=0.5) # tune level?
    # viewer.add_surface((verts, faces), name='Gt Skeleton Mesh', shading='smooth')

    viewer.dims.ndisplay = 3
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'

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
