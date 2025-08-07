from __future__ import annotations
from collections.abc import Generator
from typing import List, Any
import torch
from torch.utils.data import Sampler

class DeepONetSampler(Sampler):
    """
    A custom PyTorch Sampler for the DeepONetDataset.

    This sampler generates batches of indices tailored for DeepONet models. It provides
    control over the batch sizes for both the branch (function) and trunk (coordinate)
    inputs, enabling flexible training strategies, such as using all trunk samples for
    each branch sample or shuffling mini-batches of both.
    """
    def __init__(
        self,
        num_branch_samples: int,
        num_trunk_samples: int,
        branch_batch_size: int,
        trunk_batch_size: int | None,
        shuffle: bool = True,
    ):
        """
        Initializes the DeepONetSampler.

        Args:
            num_branch_samples (int): The total number of samples for the branch network.
            num_trunk_samples (int): The total number of samples for the trunk network.
            branch_batch_size (int): The number of branch samples to include in each batch.
            trunk_batch_size (int | None): The number of trunk samples to include in each
                                           batch. If None, all trunk samples will be used
                                           for each branch batch.
            shuffle (bool, optional): If True, shuffles the branch and trunk indices at
                                      the start of each iteration. Defaults to True.
        """
        self.num_branch_samples = num_branch_samples
        self.num_trunk_samples = num_trunk_samples
        self.branch_batch_size = branch_batch_size
        self.trunk_batch_size = trunk_batch_size
        self.shuffle = shuffle

    def __iter__(self)-> Generator[List[tuple[List[Any], List[Any] | None]], Any, None]:
        """
        Generates batches of indices for the branch and trunk data.

        This method yields a list containing a single tuple. The tuple consists of
        a list of branch indices and a list of trunk indices. If `trunk_batch_size`
        is None, the second element of the tuple is also None, indicating that
        all trunk samples should be used.

        Returns:
            Generator[List[tuple[List[Any], List[Any] | None]], Any, None]: A generator
            that yields batches of indices.
        """
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
        """
        Returns the total number of batches in one iteration.

        Returns:
            int: The total number of batches.
        """
        if self.trunk_batch_size is not None:
            return (self.num_branch_samples // self.branch_batch_size) * (self.num_trunk_samples // self.trunk_batch_size)
        else:
            return (self.num_branch_samples // self.branch_batch_size)