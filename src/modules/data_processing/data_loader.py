from __future__ import annotations
import numpy as np
import torch
import pathlib
from typing import Any
from collections.abc import Mapping
from .deeponet_transformer import DeepONetTransformPipeline


def slice_data(data: Mapping[str, np.ndarray],
               feature_keys: list[str],
               target_keys: list[str],
               split_indices: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
    branch_label = feature_keys[0]
    branch_splits = split_indices[0]
    trunk_label = feature_keys[1]
    trunk_splits = split_indices[1]
    target_labels = target_keys
    branch_data = data[branch_label]
    trunk_data = data[trunk_label]
    target_data = {key: data[key] for key in target_labels}

    split_target_data = {key: val[branch_splits][:, trunk_splits]
                         for key, val in target_data.items()}
    split_branch_data = branch_data[branch_splits]
    split_trunk_data = trunk_data[trunk_splits]
    split_data = {
        branch_label: split_branch_data,
        trunk_label: split_trunk_data,
        **split_target_data
    }
    return split_data


def get_split_data(data: Any,
                   split_indices: dict[str, np.ndarray],
                   features_keys: list[str],
                   targets_keys: list[str]) -> tuple[dict[str, np.ndarray[Any, Any]], ...]:

    branch_key = features_keys[0]
    trunk_key = features_keys[1]

    train_indices = (split_indices[f'{branch_key.upper()}_train'],
                     split_indices[f'{trunk_key.upper()}_train'])
    val_indices = (split_indices[f'{branch_key.upper()}_val'],
                   split_indices[f'{trunk_key.upper()}_train'])
    test_indices = (split_indices[f'{branch_key.upper()}_test'],
                    split_indices[f'{trunk_key.upper()}_test'])

    train_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=train_indices
    )

    val_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=val_indices
    )

    test_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=test_indices
    )

    return train_data, val_data, test_data


def get_transformed_data(data: Any, features_keys: list[str], targets_keys: list[str], transform_pipeline: DeepONetTransformPipeline) -> dict[str, Any]:
    transformed_data = {
        features_keys[0]: transform_pipeline.transform_branch(data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(data[features_keys[1]]),
        # Preserve original outputs
        **{k: transform_pipeline.transform_target(data[k]) for k in targets_keys}
    }

    return transformed_data


def get_stats(data: dict[str, np.ndarray], keys: list[str]) -> dict[str, Any]:
    """Compute statistics for normalization."""
    stats = {}
    for key in keys:
        check = any(key in k for k in data.keys())
        if not check:
            raise KeyError(f"Key {key} not found in data.")
        else:
            stats[key] = {
                'mean': data[f'{key}_mean'],
                'std': data[f'{key}_std'],
                'min': data[f'{key}_min'],
                'max': data[f'{key}_max'],
            }
    return stats


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


def get_trained_model_params(path: str | pathlib.Path) -> dict[str, Any]:
    """
    Loads the trained model parameters from a given path.

    Args:
        path (str | pathlib.Path): Path to the file containing the model parameters.

    Returns:
        dict: A dictionary containing the model parameters.
    """
    return torch.load(path, weights_only=False)
