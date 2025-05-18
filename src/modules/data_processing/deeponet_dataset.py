from __future__ import annotations
from ast import List
import logging
import torch
import numpy as np
from typing import Any
from .deeponet_transformer import DeepONetTransformer
from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


class DeepONetDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        data: dict[str, np.ndarray],
        feature_labels: list[str],
        output_labels: list[str],
        transformer: DeepONetTransformer | None = None
    ) -> None:
        print(feature_labels, data.keys())
        self.branch_data = data[feature_labels[0]]
        self.trunk_data = data[feature_labels[1]]
        self.feature_labels = feature_labels
        self.output_labels = output_labels
        self.output_data = {}
        self.transformer = transformer

        if self.transformer:
            self.trunk_data = self.transformer.transform_xt(self.trunk_data)

        for key in self.output_labels:
            field = data[key]
            n_samples = self.branch_data.shape[0]
            self.output_data[key] = field.reshape(n_samples, -1)

    def _process_index(self, index: Any, max_length: int) -> np.ndarray:
        """Convert index (int, slice, list, etc.) to a 1D array of indices."""
        if isinstance(index, int):
            return np.array([index])
        elif isinstance(index, slice):
            return np.arange(max_length)[index]
        elif isinstance(index, (list, np.ndarray)):
            return np.asarray(index).flatten()
        elif isinstance(index, (range)):
            return np.array(list(index))
        else:
            raise ValueError(f"Unsupported index type: {type(index)}")

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        if isinstance(idx, list) and isinstance(idx[0], tuple):
            idx = idx[0] # When it's a list of tuples get the tuple (solves bug when using dataloader)
        if isinstance(idx, tuple):
            idx_0, idx_1 = idx
        else:
            idx_0 = idx
            idx_1 = None

        branch_length = len(self.branch_data)
        idx_0_processed = self._process_index(idx_0, branch_length)

        if idx_1 is not None:
            trunk_length = len(self.trunk_data)
            idx_1_processed = self._process_index(idx_1, trunk_length)
        else:
            idx_1_processed = np.arange(len(self.trunk_data))

        grid_indices = np.ix_(idx_0_processed, idx_1_processed)

        output_item = {key: self.output_data[key][grid_indices] 
                       for key in self.output_labels}
        
        branch_item = self.branch_data[idx_0_processed]
        trunk_item = self.trunk_data[idx_1_processed]

        # if self.transformer:
        #     branch_item = self.transformer.transform_xb(
        #         branch_item[np.newaxis, :], training=True
        #     ).squeeze(0)
        #     output_item = {
        #         key: self.transformer.transform_output(val, key)
        #         for key, val in output_item.items()
        #     }   

        return {
            self.feature_labels[0]: branch_item,
            self.feature_labels[1]: trunk_item,
            **output_item,
        }
