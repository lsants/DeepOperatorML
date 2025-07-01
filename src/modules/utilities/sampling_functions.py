import numpy as np


def numpy_random_open_0_1(size=None):
    samples = np.random.uniform(low=0.0, high=1.0, size=size)

    while np.any(samples == 0.0):
        zero_indices = (samples == 0.0)
        num_zeros_to_replace = np.sum(zero_indices)
        new_random_values = np.random.uniform(
            low=0.0, high=1.0, size=num_zeros_to_replace)
        samples[zero_indices] = new_random_values

    return samples


def mesh_rescaling(arr: np.ndarray, c: float) -> np.ndarray:
    return c * np.log(arr / (1 - arr))
