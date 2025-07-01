import numpy as np
import matplotlib.pyplot as plt

path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/data.npz'
raw_path = './data/raw/kelvin/kelvin_v4.npz'

data = np.load(path)
raw_data = np.load(raw_path)
var_share = 0.95

xb, xt, gu = data['xb'], data['xt'], data['g_u']
x, y, z = raw_data['x'], raw_data['y'], raw_data['z']

# plt.hist(gu[:, 0])
# plt.show()

func_samples = gu.shape[0]
domain_samples = gu.shape[1]
num_channels = gu.shape[-1]


def single_basis_pod(data: np.ndarray) -> np.ndarray:
    data_stacked = data.reshape(-1, domain_samples)
    mean = np.mean(data_stacked, axis=0, keepdims=True)
    centered = (data_stacked - mean).T

    U, S, _ = np.linalg.svd(centered, full_matrices=False)

    explained_variance_ratio = np.cumsum(
        S**2) / np.linalg.norm(S, ord=2)**2

    n_modes = (explained_variance_ratio < var_share).sum().item()

    single_basis_modes = U[:, : n_modes]
    return single_basis_modes


def multi_basis_pod(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0, keepdims=True)
    centered = (data - mean).transpose(1, 0, 2)

    centered_channels_first = centered.transpose(2, 0, 1)
    U, S, _ = np.linalg.svd(centered_channels_first, full_matrices=False)
    U = U.transpose(1, 2, 0)

    explained_variance_ratio = np.cumsum(
        S**2, axis=1).transpose(1, 0) / np.linalg.norm(S, axis=1, ord=2)**2

    n_modes = max(
        (explained_variance_ratio <= var_share).sum().item(),
        max(np.argmax(explained_variance_ratio, axis=0))
    )

    multi_basis_modes = U[:, : n_modes + 1, :].transpose(0, 2, 1)
    return multi_basis_modes


share_trunk = single_basis_pod(gu)
split_outputs = multi_basis_pod(gu)

print(share_trunk.shape)
print(split_outputs.shape)
