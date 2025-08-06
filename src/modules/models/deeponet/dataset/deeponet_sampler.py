from __future__ import annotations
from collections.abc import Generator
from typing import List, Any
import torch
from torch.utils.data import Sampler

class DeepONetSampler(Sampler):
    def __init__(
        self,
        num_branch_samples: int,
        num_trunk_samples: int,
        branch_batch_size: int,
        trunk_batch_size: int | None,
        shuffle: bool = True,
    ):
        self.num_branch_samples = num_branch_samples
        self.num_trunk_samples = num_trunk_samples
        self.branch_batch_size = branch_batch_size
        self.trunk_batch_size = trunk_batch_size
        self.shuffle = shuffle

    def __iter__(self)-> Generator[List[tuple[List[Any], List[Any] | None]], Any, None]:
        branch_indices = torch.randperm(self.num_branch_samples) if self.shuffle else torch.arange(self.num_branch_samples)
        trunk_indices = torch.arange(self.num_trunk_samples)

        for i in range(0, self.num_branch_samples, self.branch_batch_size):
            branch_batch = branch_indices[i : i + self.branch_batch_size].tolist()

            if self.trunk_batch_size is not None:
                trunk_shuffled = torch.randperm(self.num_trunk_samples) if self.shuffle else trunk_indices

                for j in range(0, self.num_trunk_samples, self.trunk_batch_size):
                    trunk_batch = trunk_shuffled[j : j + self.trunk_batch_size].tolist()
                    yield [(branch_batch, trunk_batch)]
            else:
                yield [(branch_batch, None)]

    def __len__(self) -> int:
        if self.trunk_batch_size is not None:
            return (self.num_branch_samples // self.branch_batch_size) * (self.num_trunk_samples // self.trunk_batch_size)
        else:
            return (self.num_branch_samples // self.branch_batch_size)