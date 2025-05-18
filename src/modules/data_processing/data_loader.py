from __future__ import annotations
import torch
import numpy as np
import numpy.typing as npt 
from .deeponet_dataset import DeepONetDataset
from collections.abc import Mapping

def slice_dataset(data: Mapping[str, np.ndarray], 
                  feature_labels: list[str], 
                  target_labels: list[str], 
                  splits: dict[str, list[int]]) -> tuple[dict[str, np.ndarray], ...]:
    
    if len(feature_labels) != 2:
        raise ValueError("feature_labels must contain exactly two elements: branch and trunk labels.")
    branch_label, trunk_label = feature_labels
    for label in feature_labels:
        if label not in data:
            raise KeyError(f"Feature '{label}' not found in `data`.")
    required_splits = [f"{label.upper()}_{split}" for label in feature_labels for split in ["TRAIN", "VAL", "TEST"]]
    for split_key in required_splits:
        if split_key not in splits:
            raise KeyError(f"Split key '{split_key}' missing in `splits`.")
    
    branch_label = feature_labels[0]
    trunk_label = feature_labels[1]
    branch_data = data[branch_label]
    trunk_data = data[trunk_label]
    output_data = {key: data[key] for key in data if key in target_labels}

    train_branch_indices = splits[f'{branch_label.upper()}_TRAIN']
    val_branch_indices = splits[f'{branch_label.upper()}_VAL']
    test_branch_indices = splits[f'{branch_label.upper()}_TEST']

    train_trunk_indices = splits[f'{trunk_label.upper()}_TRAIN']
    val_trunk_indices = splits[f'{trunk_label.upper()}_VAL']
    test_trunk_indices = splits[f'{trunk_label.upper()}_TEST']

    train_output_data = {key: val[train_branch_indices][:, train_trunk_indices] for key, val in output_data.items()}
    train_data = {
        branch_label: branch_data[train_branch_indices],
        trunk_label: trunk_data[train_trunk_indices],
        **train_output_data
    }

    val_output_data = {key: val[val_branch_indices][:, val_trunk_indices] for key, val in output_data.items()}
    val_data = {
        branch_label: branch_data[val_branch_indices],
        trunk_label: trunk_data[val_trunk_indices],
        **val_output_data
        }

    test_output_data = {key: val[test_branch_indices][:, test_trunk_indices] for key, val in output_data.items()}
    test_data = {
        branch_label: branch_data[test_branch_indices],
        trunk_label: trunk_data[test_trunk_indices],
        **test_output_data
        }

    return train_data, val_data, test_data

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