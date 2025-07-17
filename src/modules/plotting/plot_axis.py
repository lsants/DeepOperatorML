import numpy as np
import matplotlib.pyplot as plt
from ..utilities.plot_utils import format_param


def plot_axis(coords: dict[str, np.ndarray],
              truth_field: np.ndarray,
              pred_field: np.ndarray,
              param_map: dict[str, float],
              target_labels: list[str],
              rotated=False):
    """
    Plots line comparisons along z-axis and free surface
    for the predicted and true fields by plotting 
    real and imaginary parts separately.

    Args:
        coords (dict): Dictionary of coordinate arrays. For example, for a Cartesian system,
            e.g. {'x': x_array, 'y': y_array, 'z': z_array}.
        truth_field (ndarray): Ground truth field in flattened trunk format.
        pred_field (ndarray): Predicted field in flattened trunk format.
        param_value: Parameter value(s) to display in the title.
        rotated (bool): If True, swap axes in the line plot (useful for certain cases).
    Returns:
        fig: A matplotlib Figure object.
    """
    # Format the parameter value.

    dims = list(coords.keys())
    if len(dims) < 2:
        raise ValueError(
            "At least two coordinate arrays are required for axis plotting.")

    dim1, dim2 = dims[:2]

    # If no specific axis is provided, plot line comparisons for both of the first two dimensions.
    # Here we extract two slices: one along the first dimension (fix the second at its minimum)
    # and one along the second dimension (fix the first at its minimum).
    sorted_keys = sorted(coords.keys())
    dim1, dim2 = sorted_keys[:2]
    coord1 = coords[dim1]
    coord2 = coords[dim2]

    truth_field = truth_field[:, truth_field.shape[1] // 2:, :]
    pred_field = pred_field[:, pred_field.shape[1] // 2:, :]
    truth_mesh = truth_field[0, ...] + 1j * truth_field[1, ...]
    pred_mesh = pred_field[0, ...] + 1j * pred_field[1, ...]

    # For the dim1 profile, fix dim2 at its minimum.
    slice_idx_dim2 = np.argmin(coord2)
    line_dim1_truth = truth_mesh[:, slice_idx_dim2]
    line_dim1_pred = pred_mesh[:, slice_idx_dim2]
    # For the dim2 profile, fix dim1 at its minimum.
    slice_idx_dim1 = np.argmin(coord1)
    line_dim2_truth = truth_mesh[slice_idx_dim1, :]
    line_dim2_pred = pred_mesh[slice_idx_dim1, :]

    # Set up labels.
    x_label_dim1 = dim1
    x_label_dim2 = dim2
    abs_label = r'$|\vec{u}|$'
    param_name, = param_map.keys()
    param_val, = param_map.values()

    fig, ax = plt.subplots(2, 3, figsize=(13, 8), sharex=True)

    # For dim1 profile, plot real and imaginary parts.
    ax[0, 0].plot(line_dim1_truth.real, coord1, '.-k', label='Label (real)')
    ax[0, 0].plot(line_dim1_pred.real, coord1, 'xr', label='Pred (real)')
    ax[0, 0].set_xlabel(f"{target_labels[0]}", fontsize=14)
    ax[0, 0].set_ylabel(x_label_dim1, fontsize=14)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title(f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].legend()

    ax[0, 1].plot(line_dim1_truth.imag, coord1, '.-k', label='True')
    ax[0, 1].plot(line_dim1_pred.imag, coord1, 'xr',   label='Prediction')
    ax[0, 1].set_xlabel(f"{target_labels[1]}", fontsize=14)
    ax[0, 1].set_ylabel(x_label_dim1, fontsize=14)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title(f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].legend()

    ax[0, 2].plot(np.abs(line_dim1_truth), coord1, '.-k', label='True')
    ax[0, 2].plot(np.abs(line_dim1_pred), coord1, 'xr', label='Absolute Error')
    ax[0, 2].set_xlabel(abs_label, fontsize=14)
    ax[0, 2].set_ylabel(x_label_dim1, fontsize=14)
    ax[0, 2].invert_yaxis()
    ax[0, 2].set_title(f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[0, 2].set_aspect('equal', adjustable='box')
    ax[0, 2].legend()

    # For dim2 profile.
    ax[1, 0].plot(line_dim2_truth.real, coord2, '.-k', label='True')
    ax[1, 0].plot(line_dim2_pred.real, coord2, 'xr', label='Prediction')
    ax[1, 0].set_xlabel(f"{target_labels[0]}", fontsize=14)
    ax[1, 0].set_ylabel(x_label_dim2, fontsize=14)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title(f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].legend()

    ax[1, 1].plot(line_dim2_truth.imag, coord2, '.-k', label='True')
    ax[1, 1].plot(line_dim2_pred.imag, coord2, 'xr', label='Prediction')
    ax[1, 1].set_xlabel(f"{target_labels[1]}", fontsize=14)
    ax[1, 1].set_ylabel(x_label_dim2, fontsize=14)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title(f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].legend()

    ax[1, 2].plot(np.abs(line_dim2_truth), coord2, '.-k', label='True')
    ax[1, 2].plot(np.abs(line_dim2_pred), coord2, 'xr', label='Prediction')
    ax[1, 2].set_xlabel(abs_label, fontsize=14)
    ax[1, 2].set_ylabel(x_label_dim2, fontsize=14)
    ax[1, 2].invert_yaxis()
    ax[1, 2].set_title(f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}", fontsize=15)
    # ax[1, 2].set_aspect('equal', adjustable='box')
    ax[1, 2].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
