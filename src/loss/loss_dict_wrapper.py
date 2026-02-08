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
        self.names = [name, 'loss']

    def forward(self, **batch):
        loss = self.loss(**batch)

        if isinstance(loss, dict):
            result = {self.name: loss['loss']}
            result.update(loss)
            return result
        else:
            return {
                self.name: loss,
                'loss': loss
            }
