import torch
import logging
import numpy as np
from .preprocessing import meshgrid_to_don

class DeepONetDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, input_keys=None, output_keys=None):
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

        self.input_keys = input_keys if input_keys is not None else ['input_functions', 'coordinates']

        if self.input_keys[0] not in data:
            raise ValueError(f"Branch input key '{self.input_keys[0]}' not found in data.")
        if self.input_keys[1] not in data:
            raise ValueError(f"Trunk input key '{self.input_keys[1]}' not found in data.")
        
        # ------------- Branch setting ---------------
        branch_candidate = data[self.input_keys[0]]
        if not isinstance(branch_candidate, np.ndarray):
            self.branch = meshgrid_to_don(*branch_candidate)
        else:
            self.branch = branch_candidate
            if self.branch.ndim == 1:
                self.branch = self.branch.reshape(len(self.branch), -1)
        
        # ------------- Trunk setting ---------------
        trunk_candidate = data[self.input_keys[1]]
        print("AAAAAAAAAAAAAAAA", (trunk_candidate))
        if not isinstance(trunk_candidate, np.ndarray):
            print("BBBBBBBBBBBBBBBBBBBBB", meshgrid_to_don(*trunk_candidate).shape)
            self.trunk = meshgrid_to_don(*trunk_candidate)
        else:
            print("CCCCCCCCCCCCCCCCC", meshgrid_to_don(*trunk_candidate).shape)
            self.trunk = trunk_candidate
        print("AAAAAAAAAAAAAAAA", self.trunk.shape)
        
        
        # ----------- Output setting ---------------
        if output_keys is None:
            raise ValueError("output_keys must be provided and match keys in data.")
        else:
            for key in output_keys:
                if key not in data:
                    raise ValueError(f"Output key '{key}' not found in data.")
            self.output_keys = output_keys

        num_samples = self.branch.shape[0]
        self.displacement_fields = {}
        for key in self.output_keys:
            field = data[key]
            self.displacement_fields[key] = field.reshape(num_samples, -1)
        
        self.transform = transform
        self.n_outputs = len(self.output_keys)

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        branch_input = self.branch[idx]
        trunk_input = self.trunk
        outputs = {key: self.displacement_fields[key][idx] for key in self.output_keys}

        if self.transform:
            branch_input = self.transform(branch_input)
            trunk_input = self.transform(trunk_input)
            outputs = {key: self.transform(val) for key, val in outputs.items()}

        return {'xb': branch_input, 'xt': trunk_input, **outputs, 'index': idx}

    def get_trunk(self):
        return self.transform(self.trunk) if self.transform else self.trunk
