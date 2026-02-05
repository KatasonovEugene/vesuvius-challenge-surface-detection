import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, ensemble_models):
        super().__init__()
        self.ensemble_models = nn.ModuleList(ensemble_models)

    def get_ensemble_model(self, idx):
        return self.ensemble_models[idx]

    def forward(self, volume, **batch):
        outputs = []
        for model in self.ensemble_models:
            logits = model(volume, **batch)['logits']
            probs = torch.softmax(logits, dim=1)
            outputs.append(probs)
        ensemble_predictions = torch.mean(torch.stack(outputs), dim=0)
        return {'probs': ensemble_predictions}
