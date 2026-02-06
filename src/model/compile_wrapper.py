import torch
import torch.nn as nn


class CompileWrapper(nn.Module):
    def __init__(self, model, input_keys=None, **compile_kwargs):
        super().__init__()
        self._orig_mod = model
        self.input_keys = input_keys
        self.compiled_model = torch.compile(model, **compile_kwargs)

    def get_inner_model(self):
        return self._orig_mod

    def forward(self, **batch):
        if self.input_keys is not None:
            batch = {key: batch[key] for key in self.input_keys if key in batch}
        return self.compiled_model(**batch)

    def state_dict(self, *args, **kwargs):
        return self._orig_mod.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self._orig_mod.load_state_dict(state_dict, strict=strict, assign=assign)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        
        orig = super().__getattribute__("_orig_mod")
        if hasattr(orig, name):
            return getattr(orig, name)

        compiled = super().__getattribute__("compiled_model")
        return getattr(compiled, name)
