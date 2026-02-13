import torch
import torch.nn.functional as F


def gaussian_blur_batch_3d(volume, sigmas):
    '''
    volume: [B, D, H, W]
    sigma: [B]
    '''

    assert volume.ndim == 4

    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.full((volume.shape[0],), sigmas, device=volume.device)
    elif sigmas.ndim == 0:
        sigmas = sigmas.expand(volume.shape[0])

    assert sigmas.shape[0] == volume.shape[0]

    size = int(2 * torch.ceil(sigmas.max() * 2) + 1)

    g = torch.exp((-(torch.arange(size, dtype=volume.dtype, device=volume.device) - size // 2)**2).unsqueeze(0) / (2 * sigmas**2).unsqueeze(1))
    g = g / g.sum(dim=1, keepdim=True)

    B, S = g.shape[0], g.shape[1]

    volume = volume.unsqueeze(0)
    volume = F.conv3d(volume, g.view(B, 1, S, 1, 1), padding=(size//2, 0, 0), groups=B)
    volume = F.conv3d(volume, g.view(B, 1, 1, S, 1), padding=(0, size//2, 0), groups=B)
    volume = F.conv3d(volume, g.view(B, 1, 1, 1, S), padding=(0, 0, size//2), groups=B)

    return volume.squeeze(0)


def gaussian_blur_3d(volume, sigma):
    '''
    volume: [D, H, W]
    '''

    assert(volume.ndim == 3)

    size = int(2 * torch.ceil(torch.tensor(sigma) * 2) + 1)

    g = torch.exp((-(torch.arange(size, dtype=volume.dtype, device=volume.device) - size // 2)**2) / (2 * sigma**2))
    g = g / g.sum()

    S = g.shape[0]

    volume = volume.unsqueeze(0).unsqueeze(0)
    volume = F.conv3d(volume, g.view(1, 1, S, 1, 1), padding=(size//2, 0, 0))
    volume = F.conv3d(volume, g.view(1, 1, 1, S, 1), padding=(0, size//2, 0))
    volume = F.conv3d(volume, g.view(1, 1, 1, 1, S), padding=(0, 0, size//2))

    return volume.squeeze(0).squeeze(0)