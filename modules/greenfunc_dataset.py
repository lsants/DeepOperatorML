import torch
import numpy as np
from .preprocessing import meshgrid_to_trunk, trunk_to_meshgrid

class GreenFuncDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform=None):
        data = np.load(path_to_data)
        self.xb = data['xb']
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
            freq, xt, disp_field_real, disp_field_imag = list(map(self.transform,(
                                                freq, 
                                                self.xt, 
                                                disp_field_real,
                                                disp_field_imag)
                                                )
                                            )

        sample = {'xb': freq,
                  'xt': xt, 
                  'g_u_real': disp_field_real, 
                  'g_u_imag': disp_field_imag}

        return sample
    
    def get_trunk_normalization_params(self):
        r, z = trunk_to_meshgrid(self.xt)
        min_max_params = np.array([[r.min(), z.min()],
                                   [r.max(), z.max()]])
        if self.transform:
            self.transform(min_max_params)
        
        min_max_params = [i.tolist() for i in min_max_params]

        return min_max_params