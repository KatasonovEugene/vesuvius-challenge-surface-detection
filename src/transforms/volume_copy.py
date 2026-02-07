from torch import nn


class VolumeCopy(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, volume, **batch):
        result = {self.name: volume.clone()}
        return result
