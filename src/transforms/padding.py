import numpy as np
from torch import nn


class PaddingToSize(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, volume, gt_mask=None, gt_skel=None, teacher_probs=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(f'PaddingToSize: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        if volume.shape[1] >= self.size[0] and volume.shape[2] >= self.size[1] and volume.shape[3] >= self.size[2]:
            result = {'volume': volume}
            if gt_mask is not None: 
                result['gt_mask'] = gt_mask
            if gt_skel is not None: 
                result['gt_skel'] = gt_skel
            if teacher_probs is not None:
                result['teacher_probs'] = teacher_probs
            return result

        pad_size_d = max(self.size[0] - volume.shape[1], 0)
        pad_size_h = max(self.size[1] - volume.shape[2], 0)
        pad_size_w = max(self.size[2] - volume.shape[3], 0)

        pad = lambda x : np.pad(x, ((0, 0), (0, pad_size_d), (0, pad_size_h), (0, pad_size_w)), mode='constant')

        result = {'volume': pad(volume)}
        if gt_mask is not None: 
            result['gt_mask'] = pad(gt_mask)
        if gt_skel is not None: 
            result['gt_skel'] = pad(gt_skel)
        if teacher_probs is not None:
            result['teacher_probs'] = pad(teacher_probs)
        return result
