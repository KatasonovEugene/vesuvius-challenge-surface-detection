import torch
import torch.nn.functional as F

import numpy as np
import napari

import torch
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import pyvista as pv

import scipy.ndimage as ndi

def fit_surface(volume, degree=8, grid_size = 100,
    bbox_border=3,):
    """
    Fits a PCA-aligned polynomial surface and trims it to the data extent.
    - degree: Higher = more detail (try 8-12).
    """
    volume = torch.from_numpy(volume)#.cuda()
    device = volume.device

    # 1. Extract and Center Data
    point_zyx = torch.nonzero(volume).to(device).to(torch.float64)
    if point_zyx.shape[0] < 10: return

    mean = point_zyx.mean(dim=0)
    centered = point_zyx - mean

    # 2. PCA Rotation
    U, S, V = torch.pca_lowrank(centered, q=3)
    pca_zyx = centered @ V

    # Normalization for math stability
    x_l, y_l = pca_zyx[:, 0], pca_zyx[:, 1]
    x_scale, y_scale = x_l.abs().max(), y_l.abs().max()

    # 3. Solve Polynomial
    A_list = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            A_list.append(((x_l / x_scale) ** i) * ((y_l / y_scale) ** j))
    A = torch.stack(A_list, dim=1)

    lam = 1e-2 #1e-4
    coeffs = torch.linalg.solve(
        A.T @ A + lam * torch.eye(A.shape[1], device=device, dtype=torch.float64),
        A.T @ pca_zyx[:, 2].unsqueeze(1)
    )

    # 4. Create Evaluation Grid (larger than data to ensure we hit the box edges)
    gs = grid_size
    gx = torch.linspace(x_l.min() * 1.5, x_l.max() * 1.5, gs, device=device, dtype=torch.float64)
    gy = torch.linspace(y_l.min() * 1.5, y_l.max() * 1.5, gs, device=device, dtype=torch.float64)
    grid_xl, grid_yl = torch.meshgrid(gx, gy, indexing='ij')

    # Compute local Z
    A_grid_list = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            A_grid_list.append(((grid_xl.flatten() / x_scale) ** i) * ((grid_yl.flatten() / y_scale) ** j))
    grid_zl = (torch.stack(A_grid_list, dim=1) @ coeffs).reshape(gs, gs)

    # 5. Transform back to original volume space
    grid_point = torch.stack([grid_xl.flatten(), grid_yl.flatten(), grid_zl.flatten()], dim=-1)
    grid_zyx = (grid_point @ V.T) + mean

    # --- restict to orginal sapce
    # Global coords: Index 0=Z, 1=Y, 2=X
    gz = grid_zyx[:, 0].reshape(gs, gs)
    gy = grid_zyx[:, 1].reshape(gs, gs)
    gx = grid_zyx[:, 2].reshape(gs, gs)

    D,H,W = volume.shape
    B=bbox_border
    out_of_bound = (gx < B) | (gx > W-B-1) | (gy < B) | (gy > H-B-1) | (gz < B) | (gz > D-B-1)
    mask = (out_of_bound).cpu().numpy()
    gx, gy, gz = gx.cpu().numpy(), gy.cpu().numpy(), gz.cpu().numpy()

    # apply the trim mask
    gz[mask] = np.nan
    gy[mask] = np.nan
    gx[mask] = np.nan

    ########################################################3
    valid = np.isfinite(gz)  # or support_mask before applying nan
    lab, n = ndi.label(valid)  # 4-connectivity by default in 2D
    if n > 1:
        counts = np.bincount(lab.ravel())  #this is not correct. we should drop part that doesn't contains any datapoint instead
        counts[0] = 0
        keep = counts.argmax()
        drop = lab != keep
        gz[drop] = np.nan
        gy[drop] = np.nan
        gx[drop] = np.nan

    return gz, gy, gx

##################################################################
def _build_surface_meshes(surfaces):
    meshes = []
    for gz, gy, gx in surfaces:
        if gz is None:
            continue
        N = gx.shape[0]
        verts = np.column_stack([
            gx.reshape(-1),
            gy.reshape(-1),
            gz.reshape(-1),
        ]).astype(np.float32)
        faces = []
        for i in range(N - 1):
            for j in range(N - 1):
                v0 = i * N + j
                v1 = v0 + 1
                v2 = v0 + N
                v3 = v2 + 1
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])
        faces = np.asarray(faces, dtype=np.int64)
        valid = np.isfinite(verts).all(axis=1)
        if not valid.all():
            face_mask = valid[faces].all(axis=1)
            faces = faces[face_mask]
            if faces.size == 0:
                continue
            # Remap vertex indices after dropping invalid vertices.
            old_to_new = -np.ones(verts.shape[0], dtype=np.int64)
            old_to_new[valid] = np.arange(valid.sum(), dtype=np.int64)
            faces = old_to_new[faces]
            verts = verts[valid]
        if faces.size == 0 or verts.size == 0:
            continue
        meshes.append((verts, faces))
    return meshes


def _show_fit_result_pyvista(prob_np, surfaces, threshold=0.5):
    meshes = []
    for verts, faces in _build_surface_meshes(surfaces):
        faces_pv = np.column_stack([
            np.full((faces.shape[0], 1), 3, dtype=np.int64),
            faces,
        ]).reshape(-1)
        surface = pv.PolyData(verts, faces_pv)
        surface = surface.clean()
        meshes.append(surface)

    D, H, W = prob_np.shape
    grid = pv.ImageData()
    grid.dimensions = (W + 1, H + 1, D + 1)  # (X,Y,Z)+1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    # IMPORTANT: convert (Z,Y,X) -> (X,Y,Z) before flatten(F)
    prob_np = prob_np * 0.8
    prob_np[0, 0, 0] = 1
    labels_xyz = np.transpose(prob_np, (2, 1, 0))  # (W,H,D) == (X,Y,Z)
    grid.cell_data["labels"] = labels_xyz.flatten(order="F")

    # axis
    axes = np.zeros((D, H, W))
    axes[:, 0, 0] = 1  # z axis
    axes[0, :, 0] = 1
    axes[0, 0, :] = 1
    axes[0, :, -1] = 1
    axes[0, -1, :] = 1
    axes = pv.wrap(axes)

    p = pv.Plotter()
    p.add_volume(
        axes,
        cmap="binary",
        shade=False,
    )
    p.add_volume(
        grid,
        scalars="labels",
        opacity="sigmoid",
        cmap="reds",
        shade=False,
    )
    for surface in meshes:
        p.add_mesh(
            surface,
            color="gray",
            opacity=0.8,
            show_edges=True,
            edge_color="black",
        )
    p.show()


def _show_fit_result_napari(prob_np, surfaces, threshold=0.5):
    # Napari expects (Z, Y, X)
    volume = np.clip(prob_np, 0.0, 1.0)
    thr = float(np.clip(threshold, 0.0, 1.0))
    prob_hi = np.clip((volume - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
    mask = (volume > thr).astype(np.uint8)
    meshes = _build_surface_meshes(surfaces)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        volume,
        name="prob",
        colormap="inferno",
        opacity=0.25,
        blending="additive",
        rendering="mip",
        contrast_limits=(0.0, 1.0),
    )
    viewer.add_image(
        prob_hi,
        name="prob_hi",
        colormap="magma",
        opacity=0.45,
        blending="additive",
        rendering="mip",
        contrast_limits=(0.0, 1.0),
    )
    viewer.add_labels(
        mask,
        name="mask",
        opacity=0.35,
    )
    for idx, (verts, faces) in enumerate(meshes):
        viewer.add_surface(
            (verts, faces),
            name=f"surface_{idx}",
            colormap="gray",
            opacity=0.85,
        )
    napari.run()


def show_fit_result(prob_np, surfaces, threshold=0.5):
    try:
        _show_fit_result_napari(prob_np, surfaces, threshold=threshold)
        return
    except Exception as e:
        print(f"Napari visualization failed with error: {e}")
        _show_fit_result_pyvista(prob_np, surfaces, threshold=threshold)


def calculate_polynomial_surface(probs, threshold=0.65, min_voxels=50):
    if probs.ndim == 4:
        probs = probs[0]

    probs = probs.detach().cpu().numpy()
    prob_np = probs.astype(np.float32)

    # find connected components on a binary mask
    mask = prob_np > threshold
    lab, n = ndi.label(mask)
    surfaces = []
    for idx in range(1, n + 1):
        comp = lab == idx
        if comp.sum() < min_voxels:
            continue
        gz, gy, gx = fit_surface(comp.astype(np.uint8), degree=12, grid_size=200)
        if gz is None:
            continue
        surfaces.append((gz, gy, gx))

    if not surfaces:
        return

    show_fit_result(prob_np, surfaces, threshold=threshold)
