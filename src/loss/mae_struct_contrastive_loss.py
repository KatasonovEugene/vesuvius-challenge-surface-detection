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

        self.names = [loss_name for loss_name in self.loss_struct.names if loss_name != 'loss']
        for loss_name in self.loss_contrastive.names:
            if loss_name != 'loss':
                self.names.append(loss_name)
        self.names.append("loss")

    def forward(self, logits, sem1, sem2, **batch):
        loss_struct = self.loss_struct(struct=logits, gt_struct=batch['volume'])
        loss_contrastive = self.loss_contrastive(sem1=sem1, sem2=sem2)

        result = {}
        result.update(loss_struct)
        result.update(loss_contrastive)

        loss = self.struct_weight * loss_struct['loss'] + self.contrastive_weight * loss_contrastive['loss']

        result.update({'loss': loss})

        return result
