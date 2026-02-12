import torch.nn as nn
from monai.networks.nets import SegResNet


class VecFieldPredictor(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        init_filters=16,
        dropout_prob=0.2,
        act_name="tanh",
    ):
        super().__init__()
        self.backbone = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            dropout_prob=dropout_prob,
        )
        self.act = nn.Identity()
        if act_name == "tanh":
            self.act = nn.Tanh()

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        vec_preds = self.backbone(volume)    # (B, 3, D, H, W)
        vec_preds = self.act(vec_preds)
        return {'vec_preds': vec_preds}
