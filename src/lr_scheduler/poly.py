from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power, min_lr, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super(self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = max(0, (1 - self.last_epoch) / self.max_epochs) ** self.power
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]
