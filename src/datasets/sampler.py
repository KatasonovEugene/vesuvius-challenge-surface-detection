from torch.utils.data import Sampler
from collections import defaultdict
import random


class SameShapeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.buckets = defaultdict(list)
        for i in range(len(dataset)):
            shape = dataset.get_shape(i)
            self.buckets[shape].append(i)

        self.batches = []
        for inds in self.buckets.values():
            if shuffle:
                random.shuffle(inds)

            for i in range(0, len(inds), batch_size):
                batch = inds[i:i+batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)

        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)