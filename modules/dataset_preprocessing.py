import torch
import numpy as np

class GreenFuncDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform=None):
        data = np.load(path_to_data)
        R_axis ,Z_axis = np.meshgrid(data['r'],data['z'])
        self.displacement_fields = data['g_u']
        self.xb = data['xb']
        self.xt = np.column_stack((R_axis.flatten(), Z_axis.flatten()))
        self.transform = transform

    def __len__(self):
        return len(self.displacement_field)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        freq = self.xb[idx]
        disp_field = self.displacement_fields[idx]



        sample = {'xb': freq, 'g_u': disp_field}
        if self.transform:
            sample = self.transform(sample)
        return sample