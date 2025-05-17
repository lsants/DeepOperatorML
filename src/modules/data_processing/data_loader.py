from __future__ import annotations
import torch
import numpy as np
from .deeponet_dataset import DeepONetDataset


def get_dataset_statistics(dataset: DeepONetDataset, keys: list[str] | None = None) -> dict[str, dict[str, float]]:
    """
    Compute min, max, mean, and std for specified keys in the dataset.

    Args:
        dataset (Dataset/Subset): Dataset to compute stats for.
        keys (list[str]): Keys to process (e.g., ['xb', 'xt', 'g_u']).

    Returns:
        dict: Statistics for each key: {'min', 'max', 'mean', 'std'}.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
    else:
        original_dataset = dataset
        indices = range(len(dataset))

    if keys is None:
        keys = ["xb", "xt"] + getattr(original_dataset, "output_keys", [])

    stats = {
        key: {
            "min": float("inf"),
            "max": -float("inf"),
            "sum": 0.0,
            "sum_sq": 0.0,
            "count": 0,
        }
        for key in keys
    }

    for idx in indices:
        sample = original_dataset[idx]
        for key in keys:
            if key == "xt":
                values = original_dataset.get_trunk()
            else:
                values = sample[key]

            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()

            flattened = values.reshape(-1)

            stats[key]["min"] = min(stats[key]["min"], np.min(flattened))
            stats[key]["max"] = max(stats[key]["max"], np.max(flattened))

            stats[key]["sum"] += np.sum(flattened)
            stats[key]["sum_sq"] += np.sum(flattened**2)
            stats[key]["count"] += len(flattened)

    for key in keys:
        count = stats[key]["count"]
        if count == 0:
            raise ValueError(f"No data for key: {key}")

        mean = stats[key]["sum"] / count
        variance = (stats[key]["sum_sq"] / count) - (mean**2)
        std = np.sqrt(variance)

        stats[key]["mean"] = mean
        stats[key]["std"] = std
        del stats[key]["sum"], stats[key]["sum_sq"], stats[key]["count"]

    return stats

def get_norm_params(
    train_dataset: dict[str, torch.utils.data.Subset], problem_params: dict[str, any]) -> dict[str, any]:
    """
    Collects normalization parameters (min/max/mean/std) for all keys.

    Args:
        train_dataset (Subset): Training dataset subset.
        problem_params (dict): Contains 'OUTPUT_KEYS'.

    Returns:
        dict: Normalization parameters for each key.
    """
    keys = ["xb", "xt"] + problem_params["OUTPUT_KEYS"]
    stats = get_dataset_statistics(train_dataset, keys)

    normalization_parameters = {}
    for key in keys:
        normalization_parameters[key] = {
            "min": stats[key]["min"],
            "max": stats[key]["max"],
            "mean": stats[key]["mean"],
            "std": stats[key]["std"],
        }

    return normalization_parameters

def don_to_meshgrid(arr: np.ndarray) -> tuple[np.ndarray]:
    """
    Recovers the original coordinate arrays from a trunk (or branch) array.
    
    Assumes that the trunk array was created via a meshgrid operation (with 'ij' indexing)
    from one or more 1D coordinate arrays. The trunk is a 2D array of shape (N, d) where d is
    the number of coordinate dimensions and N is the product of the lengths of the coordinate arrays.
    
    Returns a tuple of d 1D arrays containing the unique coordinate values for each dimension.
    For example, for a 2D case it returns (r, z); for a 3D case, (x, y, z).
    
    Args:
        arr (numpy.ndarray): Trunk array of shape (N, d).
    
    Returns:
        tuple: A tuple of d 1D numpy arrays corresponding to the coordinates.
    """
    d = arr.shape[1]
    coords = tuple(np.unique(arr[:, i]) for i in range(d))
    return coords

def meshgrid_to_don(*coords: np.ndarray) -> np.ndarray:
    """
    Generates the trunk/branch matrix for DeepONet training from given coordinate arrays.
    
    This function accepts either multiple coordinate arrays as separate arguments,
    or a single argument that is a list (or tuple) of coordinate arrays. It returns
    a 2D array where each row corresponds to one point in the Cartesian product of the
    input coordinate arrays.
    
    Examples:
        For 2D coordinates:
            xt = meshgrid_to_trunk(r_values, z_values)
        For 3D coordinates:
            xt = meshgrid_to_trunk(x_values, y_values, z_values)
        Or:
            xt = meshgrid_to_trunk([x_values, y_values, z_values])
    
    Args:
        *coords: One or more 1D numpy arrays representing coordinate values.
    
    Returns:
        numpy.ndarray: A 2D array of shape (N, d) where d is the number of coordinate arrays
                       and N is the product of the lengths of these arrays.
    """

    if len(coords) == 1 and isinstance(coords[0], (list, tuple)):
        coords = coords[0]
    
    meshes = np.meshgrid(*coords, indexing='ij')

    data = np.column_stack([m.flatten() for m in meshes])
    return data