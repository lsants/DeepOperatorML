import torch
import logging
import numpy as np
from .preprocessing import meshgrid_to_don

logger = logging.getLogger(__name__)

class DeepONetDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, output_keys=None):
        """
        Args:
            data (dict): Dictionary containing the data.
                It must include:
                  - Branch inputs under the key specified by input_keys[0].
                  - Trunk inputs under the key specified by input_keys[1] OR raw coordinate arrays 
                    under the key "trunk_coords". If the provided trunk data is not in the correct format, 
                    the raw coordinate arrays will be used to generate the trunk using meshgrid_to_don.
                  - Target outputs under keys specified in output_keys.
            transform (callable, optional): Transformation applied to all fields.
            input_keys (list of str): List of keys for input fields. For example: ['xb', 'xt'].
            output_keys (list of str): List of keys for output fields. These keys must exist in data.
        Raises:
            ValueError: If any required key is missing or if the outputs in data do not match the provided output_keys.
        """

        self.branch = data['xb']
        self.trunk = data['xt']
        self.output_keys = output_keys

        logger.info(f"\nShape of xb: {self.branch.shape}")
        logger.info(f"\nShape of xt: {self.trunk.shape}")
        
        # ----------- Output setting ---------------
        if self.output_keys is None:
            raise ValueError("output_keys must be provided and match keys in data.")
        else:
            for key in self.output_keys:
                if key not in data:
                    raise ValueError(f"Output key '{key}' not found in data.")

        num_samples = self.branch.shape[0]
        self.outputs = {}
        for key in self.output_keys:
            field = data[key]
            logger.info(f"Shape of {key}: {field.shape}")
            self.outputs[key] = field.reshape(num_samples, -1)
            logger.info(f"Shape of {key}: {self.outputs[key].shape}")
        
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
