from torch import nn


class MSELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, struct, gt_struct, masked, **batch):
        return ((struct - gt_struct)**2 * masked).sum() / (masked.sum() + self.eps)
