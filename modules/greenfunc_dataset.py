import torch
import numpy as np
from .preprocessing import meshgrid_to_trunk

class GreenFuncDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, output_keys=None):
        """
        Args:
            data (dict): Dictionary containing 'xb', 'xt', and 'g_u'.
            transform (callable, optional): Transformation applied to all fields.
            output_keys (list of str, optional): Keys for output fields.
        """
        self.xb = data['xb']
        if self.xb.ndim == 1:
            self.xb = self.xb.reshape(len(self.xb), -1)
        self.xt = meshgrid_to_trunk(data['r'], data['z'])

        num_features = self.xb.shape[0]
        num_coordinates = self.xt.shape[0]
        
        self.displacement_fields = np.stack([
            data['g_u'].real.reshape(num_features, num_coordinates),
            data['g_u'].imag.reshape(num_features, num_coordinates)
        ], axis=-1) 
        print(f"Shape of displacement fields: {self.displacement_fields.shape}")
        self.transform = transform

        self.output_keys = output_keys or [f"g_u_{part}" for part in ['real', 'imag']]
        self.n_outputs = len(self.output_keys)

    def __len__(self):
        return len(self.xb)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Sample containing 'xb', 'xt', output fields, and 'index'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pde_parameters = self.xb[idx]
        coordinates = self.xt
        disp_field = self.displacement_fields[idx] 

        outputs = {key: disp_field[..., i] for i, key in enumerate(self.output_keys)}

        if self.transform:
            pde_parameters = self.transform(pde_parameters)
            coordinates = self.transform(coordinates)
            outputs = {key: self.transform(output) for key, output in outputs.items()}

        sample = {
            'xb': pde_parameters, 
            'xt': coordinates, 
            **outputs,  
            'index': idx 
        }

        return sample

    def get_trunk(self):
        """
        Retrieves the trunk data (xt), applying the transform if defined.

        Returns:
            Tensor or ndarray: Transformed trunk data.
        """
        return self.transform(self.xt) if self.transform else self.xt
