import torch.nn as nn
from monai.networks.nets import UNETR


class UNETRDetector(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.backbone = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        logits = self.backbone(volume)    # (B, 2, D, H, W)
        return {'logits': logits}
