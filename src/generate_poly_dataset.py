import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import generic as gnc

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

    alpha, beta, gamma, B = X.T
    y = np.expand_dims(alpha/3 * B**3 + beta/2 * B**2 + gamma * B, axis=1)
    X = X[:, :-1]
    poly_data = np.concatenate((X, y), axis=1).astype(np.float64)
    print('Fixed dataset:', '\n', poly_data[:5], '\n')

    X, y = poly_data[:, :-1], poly_data[:, -1]
    data_norm = np.zeros((len(poly_data), 5))
    data_norm[:, :-1] = gnc.normalize(X)
    data_norm[:, -1] = y

    # --------------------- Format as torch and split train-test set ---------------------
    training_data_params, val_data_params, test_data_params = gnc.split_dataset(data, seed=42)
    mean_for_normalization, std_for_normalization = np.mean(training_data_params, axis=0)[:-1], np.std(training_data_params, axis=0)[:-1]

    training_data, val_data, test_data = gnc.split_dataset(data_norm, seed=42)

    for i in [training_data, val_data, test_data]:
        i = i[:, :-1], i[:, -1]  # format for dataset

    data_norm = data_norm[:, :-1], data_norm[:, -1]  # format for dataset

    X_train, y_train = training_data[:, :-1], training_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    X_test_mean, X_test_std = np.mean(X_test, axis=0), np.std(X_test, axis=0)
    y_test_mean, y_test_std = np.mean(y_test, axis=0), np.std(y_test, axis=0)

    training_data_tensor = PolyDataset(
        training_data, transform=ToTensor())
    val_data_tensor = PolyDataset(val_data, transform=ToTensor())
    test_data_tensor = PolyDataset(test_data, transform=ToTensor())
    X, y = PolyDataset(data_norm, transform=ToTensor()).features, PolyDataset(
        data_norm, transform=ToTensor()).labels
    print(X.shape, y.shape)

    np.savez(os.path.join(data_path, 'poly_dataset.npz'), train=training_data_tensor, val=val_data_tensor, test=test_data_tensor)
