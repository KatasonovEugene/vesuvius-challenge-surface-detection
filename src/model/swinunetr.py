import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRDetector(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_v2=True,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        dropout_path_rate=0.2,
    ):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_v2=use_v2,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        segmented_volume = self.backbone(volume)                     # (B, 2, D, H, W)
        segmented_volume = segmented_volume.permute(0, 2, 3, 4, 1)   # (B, D, H, W, 2)
        return segmented_volume
