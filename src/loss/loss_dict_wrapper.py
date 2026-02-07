from torch import nn


class BaseLossDictWrapper(nn.Module):
    def __init__(
        self,
        loss,
        name
    ):
        super().__init__()
        self.loss = loss
        self.name = name

    def forward(self, **batch):
        loss = self.loss(**batch)
        return {
            self.name: loss,
            'loss': loss
        }
