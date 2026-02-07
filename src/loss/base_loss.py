import torch
from torch import nn
from src.loss.base_losses import *


class BaseLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ce_weight=0.0,
        cld_weight=0.0,
        dice_weight=0.0,
        skel_weight=0.0,
        fp_weight=0.0,
        tversky_weight=0.0,
        eps=1e-7,
        use_downsampling=False,
        iterations=5,
        tversky_alpha=0.7,
        tversky_beta=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.ce_loss = CELoss(ignore_class_ids=2) 
        self.cld_loss = ClDiceLoss(
            use_downsampling=use_downsampling,
            iterations=iterations,
            eps=eps,
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_class_ids=2,
            eps=1e-7,
        )
        self.skel_loss = SkelLoss(eps=eps)
        self.fp_loss = FPLoss(eps=eps)
        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            eps=eps
        )

        self.losses = {
            "ce_loss": (ce_weight, self.ce_loss),
            "cld_loss": (cld_weight, self.cld_loss),
            "dice_loss": (dice_weight, self.dice_loss),
            "skel_loss": (skel_weight, self.skel_loss),
            "fp_loss": (fp_weight, self.fp_loss),
            "tversky_loss": (tversky_weight, self.tversky_loss),
        }
        self.names = []
        for loss_name, (loss_weight, _) in self.losses.items():
            if loss_weight != 0.0:
                self.names.append(loss_name)
        self.names.append("loss")

    def forward(self, **batch):
        '''
        gt_mask: [B, D, H, W]
        gt_skel: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, C, D, H, W]
        '''

        if 'probs' not in batch.keys():
            batch['probs'] = torch.softmax(batch['logits'], dim=1)
            assert batch['probs'].shape[1] == self.num_classes
            assert batch['probs'].ndim == 5

        loss_results = dict()
        for loss_name, (loss_weight, loss_fn) in self.losses.items():
            if loss_weight != 0.0:
                loss_results[loss_name] = loss_weight * loss_fn(**batch)

        final_loss = 0.0
        for loss in loss_results.values():
            final_loss += loss
        loss_results['loss'] = final_loss

        return loss_results
