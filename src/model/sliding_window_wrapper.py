import torch
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference
from src.model.ensemble import Ensemble


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
        self.is_ensemble = isinstance(self.model, Ensemble)

    def get_inner_model(self):
        return self.model

    def _predictor(self, volume):
        preds = self.model(volume=volume.squeeze(1))
        if self.is_ensemble:
            return preds['probs']
        return preds['logits']

    def forward(self, **batch):
        if self.training:
            return self.model(**batch)
        else:
            with torch.no_grad():
                preds = sliding_window_inference(
                    inputs=batch['volume'].unsqueeze(1),
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self._predictor,
                    overlap=self.overlap,
                    mode=self.mode
                )
                if self.is_ensemble:
                    return {'probs': preds, 'outputs': None}
                else:
                    return {'logits': preds[:, :2], "vector_logits": preds[:, 2:], "full_logits": None, "full_vector_logits": None}

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
