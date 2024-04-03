import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)


class PolyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if type(data) != tuple:
            self.features = data[:, :-1]
            self.labels = data[:, -1]
        else:
            self.features = data[0]
            self.labels = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = np.asarray(self.labels[idx])

        sample = features, label
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample
        return torch.tensor(features, dtype=torch.float64), torch.tensor(label, dtype=torch.float64)


def generate_data(n_samples=500, fixed=0):
    # polynomial coefficients (α, β, γ, B), in the interval [0.1, 1]
    alpha_beta_gamma_B = 2 * np.random.rand(n_samples, 4) - 1
    alpha_beta_gamma_B[:, -1] = 0.1 + np.random.rand(n_samples) * 0.9
    integrals = np.zeros((n_samples, 1))

    for i in range(n_samples):
        alpha, beta, gamma, B = alpha_beta_gamma_B[i]
        integrals[i] = (alpha / 3) * B**3 + (beta / 2) * B**2 + \
            gamma * B  # Polynomial's defined integral

    return alpha_beta_gamma_B, integrals


if __name__ == '__main__':
    data_path = os.path.join(project_dir, 'data')

    n = 1000000
    np.random.seed(42)

    X, y = generate_data(n)
    poly_data = np.concatenate((X, y), axis=1)
    print('Dataset:', '\n', poly_data[:5], '\n')
    np.save(os.path.join(data_path, 'poly_data'), poly_data)

    alpha, beta, gamma, B = X.T
    y = np.expand_dims(alpha/3 * B**3 + beta/2 * B**2 + gamma * B, axis=1)
    X = X[:, :-1]
    poly_data = np.concatenate((X, y), axis=1)
    print('Fixed dataset:', '\n', poly_data[:5], '\n')
    np.save(os.path.join(data_path, 'poly_data_fixed'), poly_data)
