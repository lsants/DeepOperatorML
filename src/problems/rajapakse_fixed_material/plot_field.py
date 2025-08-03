from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from src.modules.utilities.plot_utils import format_param

plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=12)
cmap = plt.get_cmap('viridis') # # tried: 'RdBu'
plt.rc('image', cmap=cmap.name)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'

def plot_2D_field(
        coords: dict[str, np.ndarray],
        truth_field: np.ndarray,
        pred_field: np.ndarray,
        input_function_labels: list[str],
        input_function_value: list[float],
        target_labels: list[str]
    ) -> plt.Figure:
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

    if ncols == 3:
        norm_real = colors.Normalize(vmin=min(np.min(pred_field[0]), np.min(truth_field[0])),
                                     vmax=max(np.max(pred_field[0]), np.max(truth_field[0])))
        c0 = ax[0, 0].contourf(
            X, Y, pred_field[0], norm=norm_real)
        ax[0, 0].set_title(
            f"Predicted {labels[0]} {input_function_labels[0]}={param_str}")
        ax[0, 0].set_xlabel(x_label)
        ax[0, 0].set_ylabel(y_label)

        c1 = ax[0, 1].contourf(
            X, Y, truth_field[0], norm=norm_real)
        ax[0, 1].set_title(
            f"True {labels[0]} {input_function_labels[0]}={param_str}")
        ax[0, 1].set_xlabel(x_label)

        c2 = ax[0, 2].contourf(
            X, Y, (truth_field[0] - pred_field[0]) / truth_field[0])

        ax[0, 2].set_title(
            f"Relative error {labels[0]} {input_function_labels[0]}={param_str}")
        ax[0, 2].set_xlabel(x_label)
        ax[0, 2].invert_yaxis()

        c_bar_0 = fig.colorbar(c0, ax=ax[0, 0])
        c_bar_1 = fig.colorbar(c1, ax=ax[0, 1])
        c_bar_2 = fig.colorbar(c2, ax=ax[0, 2])
        # c_bar_2.ax.yaxis.set_major_formatter(PercentFormatter(1))
        # c_bar_2.set_label('%')
    else:
        field = pred_field[0] if pred_field is not None else truth_field[0]
        c0 = ax[0, 0].contourf(X, Y, field)
        ax[0, 0].set_title(
            f"{labels[0]} {input_function_labels[0]}={param_str}")
        ax[0, 0].set_xlabel(x_label)
        ax[0, 0].set_ylabel(y_label)
        fig.colorbar(c0, ax=ax[0, 0])

    if ncols == 3:
        norm_imag = colors.Normalize(vmin=min(np.min(pred_field[1]), np.min(truth_field[1])),
                                     vmax=max(np.max(pred_field[1]), np.max(truth_field[1])))
        c3 = ax[1, 0].contourf(
            X, Y, pred_field[1], norm=norm_imag)
        ax[1, 0].set_title(
            f"Predicted {labels[1]} {input_function_labels[0]}={param_str}")
        ax[1, 0].set_xlabel(x_label)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_ylabel(y_label)

        c4 = ax[1, 1].contourf(
            X, Y, truth_field[1], norm=norm_imag)
        ax[1, 1].set_title(
            f"True {labels[1]} {input_function_labels[0]}={param_str}")
        ax[1, 1].set_xlabel(x_label)
        ax[1, 1].invert_yaxis()

        c5 = ax[1, 2].contourf(
            X, Y, (truth_field[1] - pred_field[1]) / truth_field[1])
        ax[1, 2].set_title(
            f"Relative error {labels[1]} {input_function_labels[0]}={param_str}")
        ax[1, 2].set_xlabel(x_label)
        ax[1, 2].invert_yaxis()

        c_bar_3 = fig.colorbar(c3, ax=ax[1, 0])
        c_bar_4 = fig.colorbar(c4, ax=ax[1, 1])
        c_bar_5 = fig.colorbar(c5, ax=ax[1, 2])
        # c_bar_5.ax.yaxis.set_major_formatter(PercentFormatter(1))
        # c_bar_5.set_label('%')
    else:
        field = pred_field[1] if pred_field is not None else truth_field[1]
        c3 = ax[1, 0].contourf(X, Y, field)
        ax[1, 0].set_title(
            f"{labels[1]} {input_function_labels[0]}={param_str}")
        ax[1, 0].set_xlabel(x_label)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_ylabel(y_label)

        fig.colorbar(c3, ax=ax[1, 0])

    for a in np.concatenate((ax[0, :], ax[1, :])):
        a.invert_yaxis()
    ax[0, 0].invert_yaxis()
    fig.tight_layout()
    return fig
