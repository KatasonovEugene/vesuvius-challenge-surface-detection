import torch
from torch import nn
import torch.nn.functional as F
import math


class ElasticDeformation(nn.Module):
    """
    Randomly applies Elastic Deformation transform on 3D input
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob, alpha_x, alpha_y, alpha_z, sigma):
        """
        Args:
            prob (float):
                rotate is applied with given probability
            alpha (float):
                alpha parameter of elastic deformation
            sigma (float):
                sigma parameter of elastic deformation
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z
        self.sigma = sigma

        self.size = int(2 * math.ceil(self.sigma * 2) + 1)

        g = torch.exp(-(torch.arange(self.size) - self.size // 2)**2 / (2 * self.sigma**2))
        g = g / g.sum()

        self.register_buffer('gaussian_kernel', g)

    def gaussian_blur(self, volume):
        volume = volume.unsqueeze(1)

        g = self.gaussian_kernel.to(dtype=volume.dtype, device=volume.device)
        volume = F.conv3d(volume, g.view(1, 1, -1, 1, 1), padding=(self.size//2, 0, 0))
        volume = F.conv3d(volume, g.view(1, 1, 1, -1, 1), padding=(0, self.size//2, 0))
        volume = F.conv3d(volume, g.view(1, 1, 1, 1, -1), padding=(0, 0, self.size//2))

        return volume.squeeze(1)

    def forward(self, volume, gt_mask, gt_skel, loss_weights=None, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): deformated volume tensor.
            gt_mask (Tensor): deformated ground truth mask tensor.
            gt_skel (Tensor): deformated ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'ElasticDeformation: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool).view(-1, 1, 1, 1)

        B, D, H, W = volume.shape
        device = volume.device

        dx = torch.randn(B, D, H, W, device=device)
        dy = torch.randn(B, D, H, W, device=device)
        dz = torch.randn(B, D, H, W, device=device)

        dx = self.gaussian_blur(dx) * self.alpha_x * 2 / W
        dy = self.gaussian_blur(dy) * self.alpha_y * 2 / H
        dz = self.gaussian_blur(dz) * self.alpha_z * 2 / D

        z = torch.linspace(-1, 1, D, device=device, dtype=volume.dtype)
        y = torch.linspace(-1, 1, H, device=device, dtype=volume.dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=volume.dtype)
        gridz, gridy, gridx = torch.meshgrid(z, y, x, indexing="ij")

        gridz = gridz.unsqueeze(0).expand(B, -1, -1, -1)
        gridy = gridy.unsqueeze(0).expand(B, -1, -1, -1)
        gridx = gridx.unsqueeze(0).expand(B, -1, -1, -1)

        grid = torch.stack((gridx + dx, gridy + dy, gridz + dz), dim=-1)

        volume_changed = F.grid_sample(volume.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1)
        gt_mask_changed = F.grid_sample(gt_mask.unsqueeze(1).float(), grid, mode="nearest", padding_mode="border", align_corners=True).squeeze(1)
        gt_mask_changed = gt_mask_changed.round().to(dtype=gt_mask.dtype)
        if loss_weights is not None:
            loss_weights_changed =F.grid_sample(loss_weights.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1)
        else:
            loss_weights_changed = None
        if gt_skel.dtype == torch.bool:
            gt_skel_changed = F.grid_sample(gt_skel.unsqueeze(1).float(), grid, mode="nearest", padding_mode="border", align_corners=True).squeeze(1)
            gt_skel_changed = gt_skel_changed.round().to(dtype=gt_skel.dtype)
        else:
            gt_skel_changed = F.grid_sample(gt_skel.unsqueeze(1).float(), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1)

        volume = torch.where(apply_transform, volume_changed, volume)
        gt_mask = torch.where(apply_transform, gt_mask_changed, gt_mask)
        gt_skel = torch.where(apply_transform, gt_skel_changed, gt_skel)
        if loss_weights is not None:
            loss_weights = torch.where(apply_transform, loss_weights_changed, loss_weights)

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel, 'loss_weights': loss_weights}
