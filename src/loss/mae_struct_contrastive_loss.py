from torch import nn


class MAEStructContrastiveLoss(nn.Module):
    def __init__(
        self,
        loss_struct,
        loss_contrastive,
        struct_weight=1.0,
        contrastive_weight=1.0,
    ):
        super().__init__()
        self.loss_struct = loss_struct
        self.loss_contrastive = loss_contrastive
        self.struct_weight = struct_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, logits, sem1, sem2, **batch):
        loss_struct = self.loss_struct(struct=logits, gt_struct=batch['volume'])
        loss_contrastive = self.loss_contrastive(sem1=sem1, sem2=sem2)

        result = {}
        result.update(loss_struct)
        result.update(loss_contrastive)

        loss = self.struct_weight * loss_struct['loss'] + self.contrastive_weight * loss_contrastive['loss']

        result.update({'loss': loss})

        return result
