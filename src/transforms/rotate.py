import torch
import numpy as np
from torch import nn
from src.transforms.base_tta_transform import BaseTTATransform
from scipy.ndimage import affine_transform


class RandRotate90_3D(nn.Module):
    """
    Randomly Rotate 3D input (0 or 90*k degrees, where k = rand(1, 2, ..., max_k)).
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.4, possible_k=(1, 3), spatial_axes=(0, 1)):
        """
        Args:
            prob (float):
                rotate is applied with given probability
            possible_k (tuple):
                The possible numbers of 90-degree rotations (k).
                The actual number of rotations (k) is randomly sampled uniformly from the possible_k.
            spatial_axes (tuple):
                A tuple of two integers defining the spatial axes within whose plane the rotation occurs.
                For 3D data with axes [D, H, W] or [B, D, H, W], setting (0, 1) means rotation happens in the D/H plane
                around the W-axis (axis 2).
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.register_buffer('possible_k', torch.tensor(possible_k))
        self.max_k = max(possible_k)
        self.spatial_axes = spatial_axes

    def rotate90(self, data):
        return torch.rot90(data, k=1, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim

    def forward(self, volume, gt_mask, gt_skel, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): randomly rotated volume tensor.
            gt_mask (Tensor): randomly rotated ground truth mask tensor.
            gt_skel (Tensor): randomly rotated ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandRotate90_3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool)

        koefs_idx = torch.randint(0, len(self.possible_k), size=(volume.shape[0],))
        koefs = self.possible_k[koefs_idx]
        koefs = apply_transform * koefs

        for rotate_num in range(1, self.max_k + 1):
            apply = (koefs >= rotate_num).view(-1, 1, 1, 1)
            volume = torch.where(apply, self.rotate90(volume), volume)
            gt_mask = torch.where(apply, self.rotate90(gt_mask), gt_mask)
            gt_skel = torch.where(apply, self.rotate90(gt_skel), gt_skel)

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel}


class Rotate90_3D(BaseTTATransform):
    """
    Rotate 3D input on 90*k degrees.

    Expected input shape: [B, D, H, W] for transform
    
    Expected input shape: [B, C, D, H, W] for detransform
    """

    def __init__(self, k=3, spatial_axes=(0, 1)):
        """
        Args:
            prob (float):
                rotate is applied with given probability
            k (int):
                The number of 90-degree rotations.
            spatial_axes (tuple):
                A tuple of two integers defining the spatial axes within whose plane the rotation occurs.
                For 3D data with axes [D, H, W] or [B, D, H, W], setting (0, 1) means rotation happens in the D/H plane
                around the W-axis (axis 2).
        """
        super().__init__()

        self.k = k
        self.spatial_axes = spatial_axes

    def forward(self, *, volume, gt_mask=None, gt_skel=None, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (None or Tensor): ground truth mask tensor.
            gt_skel (None or Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): rotated volume tensor.
            gt_mask (None or Tensor): rotated ground truth mask tensor.
            gt_skel (None or Tensor): rotated ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'Rotate90_3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        volume = torch.rot90(volume, k=self.k, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim
        if gt_mask is not None:
            gt_mask = torch.rot90(gt_mask, k=self.k, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim
        if gt_skel is not None:
            gt_skel = torch.rot90(gt_skel, k=self.k, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim

        result = {'volume': volume}
        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel

        return result

    def detransform(self, gt_mask=None, gt_skel=None, **batch):
        """
        Args:
            logits (Tensor): rotated logits tensor.
            gt_mask (None or Tensor): rotated ground truth mask tensor.
            gt_skel (None or Tensor): rotated ground truth skeleton tensor.
        Returns:
            logits (Tensor): derotated logits tensor.
            gt_mask (None or Tensor): derotated ground truth mask tensor.
            gt_skel (None or Tensor): derotated ground truth skeleton tensor.
        """

        if 'probs' in batch:
            preds = batch['probs']
        else:
            preds = batch['logits']

        if preds.dim() != 5:
            raise RuntimeError(f'Rotate90_3D: input shape was not expected; input shape: {preds.shape}; expected shape: [B, C, D, H, W]')

        preds = torch.rot90(preds, k=-self.k, dims=(self.spatial_axes[0] + 2, self.spatial_axes[1] + 2)) # +1 due to batch dim
        if gt_mask is not None:
            gt_mask = torch.rot90(gt_mask, k=-self.k, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim
        if gt_skel is not None:
            gt_skel = torch.rot90(gt_skel, k=-self.k, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim

        if 'probs' in batch:
            result = {'probs': preds}
        else:
            result = {'logits': preds}

        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel

        return result


class RandInstanceSmallRotate3D(nn.Module):
    """
    Randomly rotates 3D input.

    Expected input shape: [1, D, H, W]
    """

    def __init__(self, prob=0.5, angle_z_range=(-10, 10), angle_y_range=(-10, 10), angle_x_range=(-10, 10)):
        """
        Args:
            prob (float):
                rotate is applied with given probability
            angle_range (tuple):
                The range of possible rotation angles in degrees.
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.angle_x_range = angle_x_range
        self.angle_y_range = angle_y_range
        self.angle_z_range = angle_z_range


    def get_rotation_matrix(self, angles):
        rx, ry, rz = angles

        cosx = np.cos(rx)
        sinx = np.sin(rx)
        cosy = np.cos(ry)
        siny = np.sin(ry)
        cosz = np.cos(rz)
        sinz = np.sin(rz)

        rot_x = np.array([[1, 0, 0],
                          [0, cosx, -sinx],
                          [0, sinx, cosx]])

        rot_y = np.array([[cosy, 0, siny],
                          [0, 1, 0],
                          [-siny, 0, cosy]])

        rot_z = np.array([[cosz, -sinz, 0],
                          [sinz, cosz, 0],
                          [0, 0, 1]])

        return rot_z @ rot_y @ rot_x

    def forward(self, volume, gt_mask, **batch):
        """
        Args:
            volume (numpy array): volume tensor.
            gt_mask (numpy array): ground truth mask tensor.
        Returns:
            volume (numpy array): randomly rotated volume tensor.
            gt_mask (numpy array): randomly rotated ground truth mask tensor.
        """

        if volume.ndim != 4 or volume.shape[0] != 1:
            raise RuntimeError(f'RandInstanceSmallRotate3D: input shape was not expected; input shape: {volume.shape}; expected shape: [1, D, H, W]')

        apply_transform = np.random.rand() < self.prob

        if not apply_transform:
            return {'volume': volume, 'gt_mask': gt_mask}

        angles = np.radians([
            np.random.uniform(self.angle_x_range[0], self.angle_x_range[1]),
            np.random.uniform(self.angle_y_range[0], self.angle_y_range[1]),
            np.random.uniform(self.angle_z_range[0], self.angle_z_range[1])
        ])
        rotation_matrix = self.get_rotation_matrix(angles)
        inverted_rotation_matrix = np.linalg.inv(rotation_matrix)

        center = np.array(volume.shape[1:], dtype=np.float64) / 2.0
        offset = list(center - inverted_rotation_matrix @ center)

        volume = affine_transform(
            volume[0],
            inverted_rotation_matrix,
            offset=offset,
            order=1,
            mode='constant',
            cval=0.0
        )[None]

        gt_mask = affine_transform(
            gt_mask[0],
            inverted_rotation_matrix,
            offset=offset,
            order=0,
            mode='constant',
            cval=2
        )[None]

        return {'volume': volume, 'gt_mask': gt_mask}
