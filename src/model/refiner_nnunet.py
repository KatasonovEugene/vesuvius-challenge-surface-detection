import torch
from src.model.nnunet import nnUNetDetector


class nnRefiner(nnUNetDetector):
    def forward(self, volume, probs, *args, **batch):
        volume = volume.unsqueeze(1)
        probs = probs.unsqueeze(1)
        inpt = torch.cat([volume, probs], dim=1)
        
        delta = self.backbone(inpt)    # (B, 1, D, H, W)
        probs_logits = torch.logit(probs.clamp(1e-4, 1-1e-4))
        final_probs = torch.sigmoid(probs_logits + delta)

        return {'probs': final_probs}

    def get_inner_model(self):
        return self.backbone
