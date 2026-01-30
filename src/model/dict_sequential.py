from torch import nn


class DictSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, **x):
        for m in self.modules_list:
            x = m(**x)
        return x
