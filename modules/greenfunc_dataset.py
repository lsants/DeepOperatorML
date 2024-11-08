import torch
import numpy as np
from .preprocessing import meshgrid_to_trunk, trunk_to_meshgrid

class GreenFuncDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.xb = data['xb'].reshape(-1, 1)
        self.xt = meshgrid_to_trunk(data['r'],data['z'])
        self.displacement_fields = data['g_u'].reshape(len(self.xb), -1)
        self.transform = transform

    def __len__(self):
        return len(self.displacement_fields)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        freq = self.xb[idx]
        disp_field_real = self.displacement_fields[idx].real
        disp_field_imag = self.displacement_fields[idx].imag


        if self.transform:
            freq, disp_field_real, disp_field_imag = list(map(self.transform,(
                                                freq, 
                                                disp_field_real,
                                                disp_field_imag)
                                                )
                                            )

        sample = {'xb': freq,
                  'g_u_real': disp_field_real, 
                  'g_u_imag': disp_field_imag}

        return sample
    
    def get_trunk(self):
        xt = self.xt
        if self.transform:
            xt = self.transform(xt)
        return xt
    
    def get_trunk_normalization_params(self):
        r, z = trunk_to_meshgrid(self.xt)
        min_max_params = np.array([[r.min(), z.min()],
                                   [r.max(), z.max()]])

        min_max_params = {'min' : [r.min(), z.min()],
                          'max' : [r.max(), z.max()]}
        return min_max_params