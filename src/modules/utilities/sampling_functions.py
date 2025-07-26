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


# def mesh_rescaling(arr: np.ndarray, c: float):
#     if np.any(arr <= 0) or np.any(arr >= 1):
#         raise ValueError(
#             "Input array must contain values strictly between 0 and 1.")
#     numerator = arr**2 + arr - 1
#     denominator = arr * (1 - arr)

#     return c * (numerator / denominator)

def mesh_rescaling(arr: np.ndarray, c: float) -> np.ndarray:
    return arr

# def mesh_rescaling(arr: np.ndarray, c: float) -> np.ndarray:
#     return c * np.log(arr / (1 - arr))
