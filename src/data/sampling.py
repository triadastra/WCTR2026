"""Batches that preserve the panel target-month composition."""
from collections import defaultdict

import torch
from torch.utils.data import Sampler


class TargetMonthBatchSampler(Sampler):
    """One complete month per step; shuffle months, never discard targets."""

    def __init__(self, samples, shuffle=True, generator=None):
        grouped = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[sample["target_time"]].append(index)
        self.batches = [grouped[t] for t in sorted(grouped)]
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = (torch.randperm(len(self), generator=self.generator).tolist()
                 if self.shuffle else range(len(self)))
        for index in order:
            yield self.batches[index]
