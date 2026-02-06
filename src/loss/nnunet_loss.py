import torch
from torch import nn


class nnUnetLoss(nn.Module):
    def __init__(
        self,
        base_loss,
        ds_weights=None,
    ):
        super().__init__()
        self.ds_weights = ds_weights
        self.base_loss = base_loss
        self.names = self.base_loss.names

    @staticmethod
    def _nearest_resize_int(mask, size): # mask: [B, C, D, H, W]
        D, H, W = mask.shape[2], mask.shape[3], mask.shape[4]
        D_out, H_out, W_out = size
        device = mask.device

        idx_d = torch.linspace(0, D - 1, D_out, device=device).round().long()
        idx_h = torch.linspace(0, H - 1, H_out, device=device).round().long()
        idx_w = torch.linspace(0, W - 1, W_out, device=device).round().long()

        mask = mask.index_select(2, idx_d)
        mask = mask.index_select(3, idx_h)
        mask = mask.index_select(4, idx_w)
        return mask

    def forward(self, logits, outputs, gt_mask, gt_skel, **batch):
        if outputs is None:
            return self.base_loss(logits=logits, gt_mask=gt_mask, gt_skel=gt_skel)

        n_heads = outputs.shape[1]
        if self.ds_weights is None:
            weights = [1.0 / (2**i) for i in range(n_heads)]
            total_w = sum(weights)
            self.ds_weights = [w / total_w for w in weights]

        accum_results = {}

        for i in range(n_heads):
            logits_i = outputs[:, i]

            if logits_i.shape[2:] != gt_mask.shape[1:]:
                gt_mask_i = self._nearest_resize_int(gt_mask.unsqueeze(1), size=logits_i.shape[2:]).squeeze(1)
                gt_skel_i = self._nearest_resize_int(gt_skel.unsqueeze(1), size=logits_i.shape[2:]).squeeze(1)
            else:
                gt_mask_i, gt_skel_i = gt_mask, gt_skel

            result_i = self.base_loss(logits=logits_i, gt_mask=gt_mask_i, gt_skel=gt_skel_i)

            weight = self.ds_weights[i]
            if accum_results == {}:
                accum_results = {key: value * weight for key, value in result_i.items()}
            else:
                for key in result_i:
                    accum_results[key] += result_i[key] * weight

        assert all(['loss' in name for name in accum_results])

        return accum_results
