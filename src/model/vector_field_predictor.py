import torch.nn as nn
from monai.networks.nets import SegResNet


class VectorFieldPredictor(nn.Module):
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
        vector_preds = self.backbone(volume)    # (B, 3, D, H, W)
        vector_preds = self.act(vector_preds)
        return {'vector_preds': vector_preds}
