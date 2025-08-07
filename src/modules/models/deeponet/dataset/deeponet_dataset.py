from __future__ import annotations
import logging
import torch
from torch.utils import data
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

class DeepONetDataset(data.Dataset):
    """
    A PyTorch Dataset for handling data in the format required by a DeepONet model.

    This dataset organizes data into a format suitable for training DeepONets,
    where a function is mapped to a coordinate space. It handles the reshaping
    of output data and provides a flexible `__getitem__` method to retrieve
    data for both the branch and trunk networks.
    """
    def __init__(
        self,
        data: dict[str, Any],
        feature_labels: list[str],
        output_labels: list[str],
    ) -> None:
        """
        Initializes the DeepONetDataset.

        Args:
            data (Dict[str, Any]): A dictionary containing the raw data. It should
                                   contain keys corresponding to the feature and output labels.
            feature_labels (List[str]): A list of two strings, where the first is the
                                        key for the branch network's input data (function samples)
                                        and the second is the key for the trunk network's input data
                                        (coordinate samples).
            output_labels (List[str]): A list of keys for the output fields in the data.
        """
        self.branch_data = data[feature_labels[0]]
        self.trunk_data = data[feature_labels[1]]
        self.feature_labels = feature_labels
        self.output_labels = output_labels
        self.output_data = {}

        for key in self.output_labels:
            field = data[key]
            n_func_samples = self.branch_data.shape[0]
            n_coord_samples = self.trunk_data.shape[0]
            self.output_data[key] = field.reshape(
                n_func_samples, n_coord_samples, -1)

    def _process_index(self, index: Any, max_length: int) -> np.ndarray:
        """
        Convert an index (int, slice, list, etc.) to a 1D NumPy array of indices.

        This is a helper method to handle various indexing types provided to
        `__getitem__`.

        Args:
            index (Any): The index provided, which can be an int, slice, list, etc.
            max_length (int): The maximum length of the dimension being indexed.

        Returns:
            np.ndarray: A 1D array of integer indices.

        Raises:
            ValueError: If the index type is not supported.
        """
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
        """
        Retrieves a sample from the dataset.

        The method supports single-index access (retrieving all coordinate samples for
        a single function) and tuple-based indexing (retrieving a specific subset
        of function and coordinate samples).

        Args:
            idx (Union[int, Tuple, List]): The index or indices to retrieve.
                                            - An int: Retrieves all coordinate samples for a
                                                      single function (branch sample).
                                            - A tuple: Retrieves a specific subset of the data.
                                                        The first element indexes the branch data,
                                                        and the second indexes the trunk data.
                                            - A list of tuples: Handles the case where a DataLoader
                                                                might wrap the tuple index in a list.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved branch data, trunk data,
                            the corresponding output data, and the grid indices.
        """
        if isinstance(idx, list) and isinstance(idx[0], tuple):
            idx = idx[0]
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

        return {
            self.feature_labels[0]: branch_item,
            self.feature_labels[1]: trunk_item,
            'indices': grid_indices,
            **output_item,
        }
