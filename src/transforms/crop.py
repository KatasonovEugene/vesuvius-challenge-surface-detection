import numpy as np
from torch import nn


class RandSpatialCrop3D(nn.Module):
    """
    Randomly crops 3D input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple):
                tuple of three integers defining the crop sizes on each axis
        """
        super().__init__()

        self.size = size

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        """
        Args:
            volume (numpy array): volume numpy array.
            gt_mask (numpy array): ground truth mask numpy array.
            gt_skel (numpy array): ground truth skeleton numpy array.
        Returns:
            volume (numpy array): randomly cropped volume numpy array.
            gt_mask (numpy array): randomly cropped ground truth mask numpy array.
            gt_skel (numpy array): randomly cropped ground truth skeleton numpy array.
        """

        if volume.ndim != 4:
            raise RuntimeError(f'RandSpatialCrop3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        begin = np.concatenate([
            np.random.randint(low=0, high=max(1, volume.shape[1] - self.size[0] + 1), size=(volume.shape[0],))[:, None],
            np.random.randint(low=0, high=max(1, volume.shape[2] - self.size[1] + 1), size=(volume.shape[0],))[:, None],
            np.random.randint(low=0, high=max(1, volume.shape[3] - self.size[2] + 1), size=(volume.shape[0],))[:, None],
        ], axis=1)

        dz = np.arange(self.size[0])[None, :, None, None]
        dy = np.arange(self.size[1])[None, None, :, None]
        dx = np.arange(self.size[2])[None, None, None, :] 

        z_idx = begin[:, 0][:, None, None, None] + dz
        y_idx = begin[:, 1][:, None, None, None] + dy
        x_idx = begin[:, 2][:, None, None, None] + dx

        b_idx = np.arange(volume.shape[0])[:, None, None, None]

        crop = lambda x : x[b_idx, z_idx, y_idx, x_idx]
        volume = crop(volume)
        gt_mask = crop(gt_mask)
        if gt_skel is not None:
            gt_skel = crop(gt_skel)

        result = {'volume': volume, 'gt_mask': gt_mask}
        if gt_skel is not None: 
            result['gt_skel'] = gt_skel
        return result


class HighSumCrop3D(nn.Module):
    def __init__(self, size, num_candidates=10, prefer_skeleton=True):
        super().__init__()
        self.size = tuple(size)
        self.num_candidates = int(num_candidates)
        self.prefer_skeleton = bool(prefer_skeleton)

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(f'HighSumCrop3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        B, D, H, W = volume.shape
        sd, sh, sw = self.size

        out_vol = np.empty((B, sd, sh, sw), dtype=volume.dtype)
        out_msk = np.empty((B, sd, sh, sw), dtype=gt_mask.dtype)
        out_skel = np.empty((B, sd, sh, sw), dtype=gt_skel.dtype) if gt_skel is not None else None

        for b in range(B):
            valid = (gt_mask[b] != 2)

            if gt_skel is not None and self.prefer_skeleton:
                target = (gt_skel[b] > 0) & valid
            else:
                target = (gt_mask[b] == 1) & valid

            best = None
            best_score = -1

            for _ in range(self.num_candidates):
                bz = np.random.randint(0, D - sd + 1)
                by = np.random.randint(0, H - sh + 1)
                bx = np.random.randint(0, W - sw + 1)

                score = int(target[bz:bz+sd, by:by+sh, bx:bx+sw].sum())
                if score > best_score:
                    best_score = score
                    best = (bz, by, bx)

            bz, by, bx = best

            out_vol[b] = volume[b, bz:bz+sd, by:by+sh, bx:bx+sw]
            out_msk[b] = gt_mask[b, bz:bz+sd, by:by+sh, bx:bx+sw]
            if out_skel is not None:
                out_skel[b] = gt_skel[b, bz:bz+sd, by:by+sh, bx:bx+sw]

        result = {"volume": out_vol, "gt_mask": out_msk}
        if out_skel is not None:
            result["gt_skel"] = out_skel
        return result
