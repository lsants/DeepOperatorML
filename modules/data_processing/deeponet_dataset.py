from typing import Callable, Optional
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DeepONetDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, np.ndarray], transform: Optional[Callable[..., any]]=None, output_keys: Optional[list[str]] = None):
        """
        Args:
            data (dict): Dictionary containing the data.
                It must include:
                  - Branch inputs under the 'xb' key.
                  - Trunk inputs under the 'xt' key. Must be in meshgrid format: (n_coordinate_points, n_dimensions).
                  - Target outputs under keys specified in output_keys. Each output will be in a (N_input_functions, N_coordinate_points) format.
            transform (Callable, optional): Transformation applied to all fields.
            output_keys (list of str): List of keys for output fields. These keys must exist in data (e.g 'g_u').
        Raises:
            ValueError: If any required key is missing or if the outputs in data do not match the provided output_keys.
        """

        self.branch: np.ndarray = data['xb']
        self.trunk: np.ndarray = data['xt']
        self.output_keys: Optional[list[str]] = output_keys

        logger.info(f"\nShape of xb:\t{self.branch.shape}\n")
        logger.info(f"\nShape of xt:\t{self.trunk.shape}\n")
        
        if self.output_keys is None:
            raise ValueError("output_keys must be provided and match keys in data.")
        else:
            for key in self.output_keys:
                if key not in data:
                    raise ValueError(f"Output key '{key}' not found in data.")

        n_samples = self.branch.shape[0]
        self.outputs = {}

        for key in self.output_keys:
            field = data[key]
            self.outputs[key] = field.reshape(n_samples, -1)
            logger.info(f"\nShape of {key}:\t{self.outputs[key].shape}\n")
        
        self.transform = transform
        self.n_outputs = len(self.output_keys)

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
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
