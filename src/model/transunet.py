import torch.nn as nn
from medicai.models import TransUNet


class TransUNetDetector(nn.Module):
    def __init__(
        self,
        input_shape,
        encoder_name,
        classifier_activation,
        num_classes,
    ):
        self.backbone = TransUNet(
            input_shape=input_shape,
            encoder_name=encoder_name,
            classifier_activation=classifier_activation,
            num_classes=num_classes,
        )

    def forward(self, volume, **batch):
        return self.backbone(volume)
