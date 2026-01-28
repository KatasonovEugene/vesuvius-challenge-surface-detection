import torch
from torch import nn


class StandartScale(nn.Module):
    """
    Scales input using the formula:
    output = scale * input
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): scale value used in the transform.
        """
        self.scale = scale
        super().__init__()

    def forward(self, volume):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): scaled tensor.
        """
        volume = self.scale * volume
        return volume


class ScaleIntensityRange(nn.Module):
    """
    Scales input from [a_min, a_max] to [b_min, b_max].

    Performs a linear intensity rescaling using the formula:
    output = (input - a_min) * ((b_max - b_min) / (a_max - a_min)) + b_min
    """

    def __init__(self, a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True):
        """
        Args:
            a_min (float or int): Minimum intensity value of the input data range.
            a_max (float or int): Maximum intensity value of the input data range.
            b_min (float or int): Minimum intensity value of the output data range.
            b_max (float or int): Maximum intensity value of the output data range.
            clip (bool): Whether to clip the output intensity values to the [b_min, b_max] range after scaling.
        """
        super().__init__()

        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def forward(self, volume):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): scaled tensor.
        """

        volume = (volume - self.a_min) * ((self.b_max - self.b_min) / (self.a_max - self.a_min)) + self.b_min
        if self.clip:
            volume = torch.clamp(volume, min=self.b_min, max=self.b_max)

        return {'volume': volume}

