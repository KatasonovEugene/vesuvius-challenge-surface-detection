import torch
import numpy as np
import scipy.ndimage as ndi


def build_anisotropic_struct_(z_radius: int, xy_radius: int):
    z, r = z_radius, xy_radius

    if z == 0 and r == 0:
        return None

    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct

    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct

    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct

def anisotropoc_closing(mask: np.ndarray, z_radius: int, xy_radius: int):
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct_(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)
    return mask



def hysteresis(volume: np.ndarray, t_low: float, t_high: float):
    strong = volume >= t_high
    weak   = volume >= t_low

    if not strong.any():
        return np.zeros_like(volume, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    return mask
