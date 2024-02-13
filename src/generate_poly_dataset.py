import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PolyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.features = data[0]
        self.labels = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        features = self.features[idx]
        label = np.array(self.labels[idx])

        sample = features, label
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample
        return torch.from_numpy(features), torch.from_numpy(label)


def generate_data(n_samples=500):
    # polynomial coefficients (α, β, γ, B), in the interval [0, 1]
    alpha_beta_gamma_B = np.random.rand(n_samples, 4)
    integrals = np.zeros((n_samples, 1))

    for i in range(n_samples):
        alpha, beta, gamma, B = alpha_beta_gamma_B[i]
        integrals[i] = (alpha / 3) * B**3 + (beta / 2) * B**2 + \
            gamma * B  # Polynomial's defined integral

    return alpha_beta_gamma_B, integrals


if __name__ == '__main__':
    path = os.getcwd()
    data_path = os.path.join(path, 'data')

    n = 1000000
    X, y = generate_data(n)

    poly_data = np.concatenate((X, y), axis=1)
    print(poly_data.shape)
    np.save(os.path.join(data_path, 'poly_data'), poly_data)
