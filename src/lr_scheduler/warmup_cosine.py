from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, min_end_lr, warmup_ratio, steps, last_epoch=-1):
        self.warmup_steps = int(warmup_ratio * steps)
        self.cosine_steps = steps - self.warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_end_lr = min_end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            ratio = self.last_epoch / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * ratio
        else:
            ratio = (self.last_epoch - self.warmup_steps) / self.cosine_steps
            lr = self.min_end_lr + (self.max_lr - self.min_end_lr) * (1 + np.cos(ratio * np.pi)) / 2
        return [lr for _ in self.optimizer.param_groups]
