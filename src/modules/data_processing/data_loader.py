from __future__ import annotations
import numpy as np
from collections.abc import Mapping

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
    
    split_target_data = {key: val[branch_splits][ : , trunk_splits] for key, val in target_data.items()}
    split_branch_data = branch_data[branch_splits]
    split_trunk_data = trunk_data[trunk_splits]
    split_data = {
        branch_label: split_branch_data,
        trunk_label: split_trunk_data,
        **split_target_data
    }
    return split_data

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