from torch import nn
import torch
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=3, eps=1e-7):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.kernel = torch.ones((1, 1, window_size, window_size, window_size)) / (window_size ** 3)

    def forward(self, struct, gt_struct, masked, **batch):
        mu_x = F.conv3d(struct, self.kernel, padding=self.window_size // 2, groups=1)
        mu_y = F.conv3d(gt_struct, self.kernel, padding=self.window_size // 2, groups=1)

        sigma_x = F.conv3d(struct**2, self.kernel, padding=self.window_size // 2, groups=1) - mu_x**2
        sigma_y = F.conv3d(gt_struct**2, self.kernel, padding=self.window_size // 2, groups=1) - mu_y**2
        sigma_xy = F.conv3d(struct * gt_struct, self.kernel, padding=self.window_size // 2, groups=1) - mu_x * mu_y

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        ssim_masked = (ssim_map * masked).sum() / (masked.sum() + self.eps)
        return 1.0 - ssim_masked.mean()
