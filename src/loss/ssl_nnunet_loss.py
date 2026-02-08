from torch import nn


class SSLnnUnetLoss(nn.Module):
    def __init__(
        self,
        base_loss,
        ds_weights=None,
    ):
        super().__init__()
        self.ds_weights = ds_weights
        self.base_loss = base_loss
        self.names = self.base_loss.names


    def forward(self, logits, outputs, sem1, sem2, **batch):
        if outputs is None:
            return self.base_loss(logits=logits, sem1=sem1, sem2=sem2, **batch)

        n_heads = outputs.shape[1]
        if self.ds_weights is None:
            weights = [1.0 / (2**i) for i in range(n_heads)]
            total_w = sum(weights)
            self.ds_weights = [w / total_w for w in weights]

        accum_results = {}

        for i in range(n_heads):
            logits_i = outputs[:, i]

            assert logits_i.shape[2:] == batch['volume'].shape[1:]

            sem1_i = sem1
            sem2_i = sem2
            result_i = self.base_loss(logits=logits_i, sem1=sem1_i, sem2=sem2_i, **batch)

            weight = self.ds_weights[i]
            if accum_results == {}:
                accum_results = {key: value * weight for key, value in result_i.items()}
            else:
                for key in result_i:
                    accum_results[key] += result_i[key] * weight

        assert all(['loss' in name for name in accum_results])

        return accum_results
