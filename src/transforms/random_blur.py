import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGaussianBlur3D(nn.Module):
    def __init__(self, p=0.5, num_points=(5, 7), radius=(3, 5), sigma=(0.8, 1.6)):
        super().__init__()
        self.p = p
        self.num_points = num_points
        self.radius = radius
        self.sigma = sigma

    def _g1(self, k, s, device, dtype):
        x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
        g = torch.exp(-(x * x) / (2 * s * s))
        return g / g.sum().clamp_min(1e-12)

    def forward(self, volume, gt_mask, **batch):
        if (not self.training) or torch.rand((), device=volume.device) > self.p:
            return {'volume': volume}

        x = volume.unsqueeze(0)
        d, h, w = volume.shape[1:]
        dev, dt = volume.device, volume.dtype

        idx = (gt_mask[0] == 1).nonzero(as_tuple=False)
        if idx.numel() == 0:
            return {'volume': volume}

        n = int(torch.randint(self.num_points[0], self.num_points[1] + 1, (1,), device=dev).item())
        sel = idx[torch.randint(0, idx.shape[0], (n,), device=dev)]
        r = int(torch.randint(self.radius[0], self.radius[1] + 1, (1,), device=dev).item())
        s = float(torch.empty((), device=dev).uniform_(self.sigma[0], self.sigma[1]).item())
        k = 2 * r + 1

        g = self._g1(k, s, dev, torch.float32).to(dt)
        ker = (g[:, None, None] * g[None, :, None] * g[None, None, :]).view(1, 1, k, k, k)
        blurred = F.conv3d(x, ker, padding=r)

        region = torch.zeros((d, h, w), device=dev, dtype=torch.bool)
        for z, y, x0 in sel.tolist():
            z0, z1 = max(z - r, 0), min(z + r + 1, d)
            y0, y1 = max(y - r, 0), min(y + r + 1, h)
            x0b, x1 = max(x0 - r, 0), min(x0 + r + 1, w)
            region[z0:z1, y0:y1, x0b:x1] = True

        out = torch.where(region.unsqueeze(0), blurred, x)
        return {'volume': out.squeeze(0)}
