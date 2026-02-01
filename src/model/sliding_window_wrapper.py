from typing import Any, Mapping
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference


class SlidingWindowWrapper(nn.Module):
    def __init__(
        self,
        model,
        roi_size,
        sw_batch_size=1,
        overlap=0.5,
        mode='gaussian'
    ):
        super().__init__()
        self.model = model
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode

    def _predictor(self, volume):
        return self.model(volume=volume.squeeze(1))['logits']

    def forward(self, **batch):
        if self.training:
            return self.model(**batch)
        else:
            logits = sliding_window_inference(
                inputs=batch['volume'].unsqueeze(1),
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self._predictor,
                overlap=self.overlap,
                mode=self.mode
            )
            return {'logits': logits}

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
