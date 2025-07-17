from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..utilities.plot_utils import format_param


def plot_2D_field(coords: dict[str, np.ndarray],
                  truth_field: np.ndarray,
                  pred_field: np.ndarray,
                  input_function_labels: list[str],
                  input_function_value: list[float],
                  target_labels: list[str]
                  ) -> plt.Figure:
    """
    Plots a 2D contour field (or fields) on a specified plane.
    The function accepts a dictionary of coordinate arrays and an optional dictionary of labels.
    You can choose which two coordinates to plot by specifying the key 'plot_dims' in coord_labels.
    If additional coordinates exist, the function reshapes the flattened field into a higher-dimensional
    array and slices the extra dimension at its middle index.

    If both truth_field and pred_field are provided, the function plots a 1×3 subplot arrangement
    (prediction, label, absolute error). If only one field is provided, only that field is plotted.
    If the output fields are complex and both are provided, the function plots real and imaginary parts
    in separate rows; if only one is provided, it plots the available parts.

    Additionally, if the chosen horizontal coordinate (assumed to be the first coordinate) has all positive values,
    the function mirrors it (concatenates with its negative counterpart, excluding the duplicate at zero)
    and mirrors the corresponding field.

    Args:
        coords (dict): Dictionary where keys are coordinate names (e.g. 'x', 'y', 'z')
                       and values are 1D numpy arrays.
        truth_field (ndarray, optional): Ground truth field in flattened trunk format.
        pred_field (ndarray, optional): predictedicted field in flattened trunk format.
        param_value: A parameter value (or tuple/dict) to display in the title.
        coord_labels (dict, optional): Dictionary with keys:
            - 'plot_dims': tuple of two strings indicating which coordinate keys to plot, e.g. ('x','z')
            - For each coordinate key, a label string.
            If not provided, defaults to the first two keys in coords.

    Returns:
        fig: A matplotlib Figure object.
    """

    dims = list(coords.keys())
    if len(dims) < 2:
        raise ValueError(
            "At least two coordinate arrays are required for plotting.")
    dim1, dim2 = dims[:2]

    coord1 = coords[dim1]
    coord2 = coords[dim2]

    if np.all(coord1 >= 0):
        coord1_full = np.concatenate((-np.flip(coord1), coord1))
    else:
        coord1_full = coord1

    # Create a 2D meshgrid using the full horizontal coordinate.
    X, Y = np.meshgrid(coord1_full, coord2, indexing='ij')

    param_str = format_param(param=input_function_value,
                             param_keys=input_function_labels)
    labels = target_labels

    x_label, y_label = dim1, dim2

    # If both fields are provided, plot prediction, label, and error; otherwise, plot one column.
    ncols = 3 if (
        truth_field is not None and pred_field is not None) else 1
    fig, ax = plt.subplots(2, ncols, figsize=(
        6 * ncols, 12), sharex=True, sharey=True)

    # Row 0: Real parts.
    if ncols == 3:
        norm_real = colors.Normalize(vmin=min(np.min(pred_field[0]), np.min(truth_field[0])),
                                     vmax=max(np.max(pred_field[0]), np.max(truth_field[0])))
        c0 = ax[0, 0].contourf(
            X, Y, pred_field[0], cmap='viridis', norm=norm_real)
        ax[0, 0].set_title(
            f"{labels[0]} predicted {input_function_labels[0]}={param_str}")
        ax[0, 0].set_xlabel(x_label)
        ax[0, 0].set_ylabel(y_label)

        c1 = ax[0, 1].contourf(
            X, Y, truth_field[0], cmap='viridis', norm=norm_real)
        ax[0, 1].set_title(
            f"{labels[0]} true {input_function_labels[0]}={param_str}")
        ax[0, 1].set_xlabel(x_label)

        c2 = ax[0, 2].contourf(X, Y, np.abs(
            truth_field[0] - pred_field[0]), cmap='viridis')
        ax[0, 2].set_title(
            f"Absolute error ({labels[0]}) {input_function_labels[0]}={param_str}")
        ax[0, 2].set_xlabel(x_label)
        ax[0, 2].invert_yaxis()

        fig.colorbar(c0, ax=ax[0, 0])
        fig.colorbar(c1, ax=ax[0, 1])
        fig.colorbar(c2, ax=ax[0, 2])
    else:
        field = pred_field[0] if pred_field is not None else truth_field[0]
        c0 = ax[0, 0].contourf(X, Y, field, cmap='viridis')
        ax[0, 0].set_title(
            f"{labels[0]} {input_function_labels[0]}={param_str}")
        ax[0, 0].set_xlabel(x_label)
        ax[0, 0].set_ylabel(y_label)
        fig.colorbar(c0, ax=ax[0, 0])

    # Row 1: Imaginary parts.
    if ncols == 3:
        norm_imag = colors.Normalize(vmin=min(np.min(pred_field[1]), np.min(truth_field[1])),
                                     vmax=max(np.max(pred_field[1]), np.max(truth_field[1])))
        c3 = ax[1, 0].contourf(
            X, Y, pred_field[1], cmap='viridis', norm=norm_imag)
        ax[1, 0].set_title(
            f"{labels[1]} predicted {input_function_labels[0]}={param_str}")
        ax[1, 0].set_xlabel(x_label)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_ylabel(y_label)

        c4 = ax[1, 1].contourf(
            X, Y, truth_field[1], cmap='viridis', norm=norm_imag)
        ax[1, 1].set_title(
            f"{labels[1]} true {input_function_labels[0]}={param_str}")
        ax[1, 1].set_xlabel(x_label)
        ax[1, 1].invert_yaxis()

        c5 = ax[1, 2].contourf(X, Y, np.abs(
            truth_field[1] - pred_field[1]), cmap='viridis')
        ax[1, 2].set_title(
            f"Absolute error ({labels[1]}) {input_function_labels[0]}={param_str}")
        ax[1, 2].set_xlabel(x_label)
        ax[1, 2].invert_yaxis()

        fig.colorbar(c3, ax=ax[1, 0])
        fig.colorbar(c4, ax=ax[1, 1])
        fig.colorbar(c5, ax=ax[1, 2])
    else:
        field = pred_field[1] if pred_field is not None else truth_field[1]
        c3 = ax[1, 0].contourf(X, Y, field, cmap='viridis')
        ax[1, 0].set_title(
            f"{labels[1]} {input_function_labels[0]}={param_str}")
        ax[1, 0].set_xlabel(x_label)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_ylabel(y_label)

        fig.colorbar(c3, ax=ax[1, 0])

    # Invert vertical axis for all subplots.
    for a in np.concatenate((ax[0, :], ax[1, :])):
        a.invert_yaxis()
    ax[0, 0].invert_yaxis()
    fig.tight_layout()
    return fig
