import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class SegResNetDetector(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.backbone = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            dropout_prob=dropout_prob,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        logits = self.backbone(volume)    # (B, 2, D, H, W)
        return {'logits': logits}
