import torch
import torch.nn as nn
import torch.nn.functional as F


class NNUnetMAEStructSemantic(nn.Module):
    def __init__(
        self,
        model,
        projection_hidden_dims=[512, 512, 128],
    ):
        super().__init__()
        self.model = model

        in_dim = 2 * model.get_encoder_channels()
        dims = [in_dim] + projection_hidden_dims
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        self.projection = nn.Sequential(*layers)

    def get_sem(self, volume):
        sem = self.model(volume, return_features=True)

        avg = F.adaptive_avg_pool3d(sem, 1)
        mx = F.adaptive_max_pool3d(sem, 1)

        sem = torch.cat([avg, mx], dim=1).flatten(1)
        sem = self.projection(sem)
        sem = F.normalize(sem, dim=1)
        return sem

    def forward(self, volume_struct, volume_sem1, volume_sem2, **batch):
        struct = self.model(volume_struct)
        sem1 = self.get_sem(volume_sem1)
        sem2 = self.get_sem(volume_sem2)

        return {'logits': struct['logits'], 'outputs': struct['outputs'], 'sem1': sem1, 'sem2': sem2}
