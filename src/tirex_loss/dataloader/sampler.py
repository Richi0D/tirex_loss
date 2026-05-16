import random
import numpy as np
from torch.utils.data import BatchSampler


class SameLengthBatchSampler(BatchSampler):
    def __init__(self, prediction_lengths, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # group indices by prediction length
        self.groups = {}
        for idx, s in enumerate(prediction_lengths):
            s = int(s)
            if s not in self.groups:
                self.groups[s] = []
            self.groups[s].append(idx)

    def __iter__(self):
        batches = []
        for s, indices in self.groups.items():
            indices = np.array(indices)
            if self.shuffle:
                np.random.shuffle(indices)
            # split into chunks of batch_size
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i : i + self.batch_size].tolist())
        
        if self.shuffle:
            random.shuffle(batches)  # shuffle order of batches across groups
        
        yield from batches

    def __len__(self):
        return sum(len(v) // self.batch_size for v in self.groups.values())