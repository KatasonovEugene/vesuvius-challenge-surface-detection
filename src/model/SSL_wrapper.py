import torch.nn as nn
import torch.nn.functional as F


class NNUnetMAEStructSemantic(nn.Module):
    def __init__(
        self,
        model,
        projection_hidden_dims=[512, 128],
    ):
        super().__init__()
        self.model = model

        self.projection = nn.Sequential(
            nn.Linear(model.get_encoder_channels(), projection_hidden_dims[0]),
        )
        for i in range(len(projection_hidden_dims) - 1):
            self.projection.append(nn.ReLU())
            self.projection.append(nn.Linear(projection_hidden_dims[i], projection_hidden_dims[i + 1]))

    def forward(self, volume_struct, volume_sem1, volume_sem2, **batch):
        struct = self.model(volume_struct)
        sem1 = self.model(volume_sem1, return_features=True)
        sem2 = self.model(volume_sem2, return_features=True)

        sem1 = sem1.mean(dim=(2, 3, 4))
        sem2 = sem2.mean(dim=(2, 3, 4))

        sem1 = self.projection(sem1)
        sem2 = self.projection(sem2)

        sem1 = F.normalize(sem1, dim=1)
        sem2 = F.normalize(sem2, dim=1)

        return {'logits': struct['logits'], 'outputs': struct['outputs'], 'sem1': sem1, 'sem2': sem2}
