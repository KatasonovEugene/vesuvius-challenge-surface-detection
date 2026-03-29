import torch
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference

from src.utils.model_utils import get_wrapped_ensemble, is_wrapped_ensemble


class SlidingWindowWrapper(nn.Module):
    def __init__(
        self,
        model,
        roi_size,
        sw_batch_size=1,
        overlap=0.5,
        mode='gaussian',
        output_key=None,
    ):
        super().__init__()
        self.model = model
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.is_ensemble = is_wrapped_ensemble(self.model)

        if output_key is None:
            if self.is_ensemble:
                ensemble = get_wrapped_ensemble(self.model)
                if ensemble is not None and getattr(ensemble, "ensemble_type", "probs") == "logits":
                    self.output_key = "logits"
                else:
                    self.output_key = "probs"
            else:
                self.output_key = "logits"
        else:
            self.output_key = str(output_key)

    def get_inner_model(self):
        return self.model

    def _predictor(self, volume):
        preds = self.model(volume=volume.squeeze(1))
        return preds[self.output_key]

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
                return {self.output_key: preds, 'outputs': None}

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
