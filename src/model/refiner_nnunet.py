import torch
from src.model.nnunet import nnUNetDetector


class nnRefiner(nnUNetDetector):
    def forward(self, volume, teacher_probs, *args, **batch):
        volume = volume.unsqueeze(1)
        teacher_probs = teacher_probs.unsqueeze(1)
        inpt = torch.cat([volume, teacher_probs], dim=1)
        
        delta = self.backbone(inpt)    # (B, 1, D, H, W)
        teacher_probs_logits = torch.logit(teacher_probs.clamp(1e-4, 1-1e-4))
        final_probs = torch.sigmoid(teacher_probs_logits + delta)

        return {'probs': final_probs}

    def get_inner_model(self):
        return self.backbone
