from torch import nn


class TempRenameKeys(nn.Module):
    def __init__(self, key_changes, transforms):
        super().__init__()
        self.key_changes = key_changes
        self.transforms = transforms

    def forward(self, **batch):
        for old_key, new_key in self.key_changes.items():
            batch[new_key] = batch.pop(old_key)

        for transform_name in self.transforms.keys():
            transform_result = self.transforms[transform_name](**batch)
            batch.update(transform_result)

        for new_key, old_key in reversed(list(self.key_changes.items())):
            batch[old_key] = batch.pop(new_key)

        return batch
