import torch.nn as nn


class BaseTTATransform(nn.Module):
    """
    Base class for Test-Time Augmentation transforms.
    Every TTA transform must implement:
      - forward()      : apply augmentation
      - detransform()  : invert augmentation on predictions
    """

    def forward(self, **batch):
        raise NotImplementedError()

    def detransform(self, **batch):
        raise NotImplementedError()
