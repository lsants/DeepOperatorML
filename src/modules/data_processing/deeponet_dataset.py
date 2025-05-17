from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DeepONetDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, data: dict[str, np.ndarray] , output_keys: list[str], transform: Callable[..., Any] | None = None) -> None:
        """
        Args:
            data (dict): Dictionary containing the data.
                It must include:
                  - Branch inputs under the 'xb' key.
                  - Trunk inputs under the 'xt' key. Must be in meshgrid format: (n_coordinate_points, n_dimensions).
                  - Target outputs under keys specified in output_keys. Each output will be in a (N_input_functions, N_coordinate_points) format.
            output_keys (list of str): List of keys for output fields. These keys must exist in data (e.g 'g_u').
            transform (Callable, optional): Transformation applied to all fields.
        Raises:
            ValueError: If any required key is missing or if the outputs in data do not match the provided output_keys.
        """

        self.branch: np.ndarray = data['xb']
        self.trunk: np.ndarray = data['xt']
        self.output_keys = output_keys
        
        for key in self.output_keys:
            if key not in data:
                raise ValueError(f"Output key '{key}' not found in data.")

        n_samples = self.branch.shape[0]
        self._outputs = {}

        for key in self.output_keys:
            field = data[key]
            self.outputs[key] = field.reshape(n_samples, -1)
        
        self.transform = transform
        self.n_outputs = len(self.output_keys)

    def __len__(self) -> int:
        return len(self.branch)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        branch_input = self.branch[idx]
        trunk_input = self.trunk
        outputs = {key: self.outputs[key][idx] for key in self.output_keys}

        if self.transform:
            branch_input = self.transform(branch_input)
            trunk_input = self.transform(trunk_input)
            outputs = {key: self.transform(val) for key, val in outputs.items()}

        return {'xb': branch_input, 'xt': trunk_input, **outputs, 'index': idx}

    def get_trunk(self):
        return self.transform(self.trunk) if self.transform else self.trunk
