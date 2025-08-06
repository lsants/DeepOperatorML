import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_axis(coords: dict[str, np.ndarray],
              truth_field: np.ndarray,
              pred_field: np.ndarray,
              param_map: dict[str, float],
              target_labels: list[str],
              rotated=False) -> Figure:

    dims = list(coords.keys())
    if len(dims) < 2:
        raise ValueError(
            "At least two coordinate arrays are required for axis plotting.")

    dim1, dim2 = dims[:2]
    sorted_keys = sorted(coords.keys())
    dim1, dim2 = sorted_keys[:2]
    coord1 = coords[dim1]
    coord2 = coords[dim2]

    truth_field = truth_field[:, truth_field.shape[1] // 2:, :]
    pred_field = pred_field[:, pred_field.shape[1] // 2:, :]
    truth_mesh = truth_field[0, ...] + 1j * truth_field[1, ...]
    pred_mesh = pred_field[0, ...] + 1j * pred_field[1, ...]

    slice_idx_dim2 = np.argmin(coord2)
    line_dim1_truth = truth_mesh[:, slice_idx_dim2]
    line_dim1_pred = pred_mesh[:, slice_idx_dim2]

    slice_idx_dim1 = np.argmin(coord1)
    line_dim2_truth = truth_mesh[slice_idx_dim1, :]
    line_dim2_pred = pred_mesh[slice_idx_dim1, :]

    x_label_dim1 = dim1
    x_label_dim2 = dim2
    abs_label = r'$|\mathbf{u}|$'
    param_name, = param_map.keys()
    param_val, = param_map.values()

    fig, ax = plt.subplots(2, 3, figsize=(13, 10), sharex=True)

    ax[0, 0].plot(line_dim1_truth.real, coord1, '.-k', label='True')
    ax[0, 0].plot(line_dim1_pred.real, coord1, 'xr', label='Predicted')
    ax[0, 0].set_xlabel(f"{target_labels[0]}")
    ax[0, 0].set_ylabel(x_label_dim1)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title(
        f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}")
    # ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].legend()

    ax[0, 1].plot(line_dim1_truth.imag, coord1, '.-k', label='True')
    ax[0, 1].plot(line_dim1_pred.imag, coord1, 'xr',   label='Predicted')
    ax[0, 1].set_xlabel(f"{target_labels[1]}")
    ax[0, 1].set_ylabel(x_label_dim1)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title(
        f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}")
    # ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].legend()

    ax[0, 2].plot(np.abs(line_dim1_truth), coord1, '.-k', label='True')
    ax[0, 2].plot(np.abs(line_dim1_pred), coord1, 'xr', label='Predicted')
    ax[0, 2].set_xlabel(abs_label)
    ax[0, 2].set_ylabel(x_label_dim1)
    ax[0, 2].invert_yaxis()
    ax[0, 2].set_title(
        f"({dim1}=0, {dim2}) at {param_name}={param_val:.3f}")
    # ax[0, 2].set_aspect('equal', adjustable='box')
    ax[0, 2].legend()

    # For dim2 profile.
    ax[1, 0].plot(line_dim2_truth.real, coord2, '.-k', label='True')
    ax[1, 0].plot(line_dim2_pred.real, coord2, 'xr', label='Predicted')
    ax[1, 0].set_xlabel(f"{target_labels[0]}")
    ax[1, 0].set_ylabel(x_label_dim2)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title(
        f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}")
    # ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].legend()

    ax[1, 1].plot(line_dim2_truth.imag, coord2, '.-k', label='True')
    ax[1, 1].plot(line_dim2_pred.imag, coord2, 'xr', label='Predicted')
    ax[1, 1].set_xlabel(f"{target_labels[1]}")
    ax[1, 1].set_ylabel(x_label_dim2)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title(
        f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}")
    # ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].legend()

    ax[1, 2].plot(np.abs(line_dim2_truth), coord2, '.-k', label='True')
    ax[1, 2].plot(np.abs(line_dim2_pred), coord2, 'xr', label='Predicted')
    ax[1, 2].set_xlabel(abs_label)
    ax[1, 2].set_ylabel(x_label_dim2)
    ax[1, 2].invert_yaxis()
    ax[1, 2].set_title(
        f"({dim1}, {dim2}=0) at {param_name}={param_val:.3f}")
    # ax[1, 2].set_aspect('equal', adjustable='box')
    ax[1, 2].legend()

    # fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
