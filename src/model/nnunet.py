import torch.nn as nn
from monai.networks.nets.dynunet import DynUNet


class nnUNetDetector(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_size=[[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
        strides=[[1, 1, 1],[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2]],
        deep_supr_num=1,
    ):
        super().__init__()
        self.backbone = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=features_per_stage,
            res_block=True,
            deep_supervision=True,
            deep_supr_num=deep_supr_num,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        logits = self.backbone(volume)    # (B, 2, D, H, W)
        if self.training:
            return {'logits': logits[:, 0], "outputs": logits}
        else:
            return {'logits': logits, "outputs": None}
