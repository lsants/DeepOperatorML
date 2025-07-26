from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.modules.utilities.plot_utils import format_param
from matplotlib.ticker import PercentFormatter

plt.rc('font', family='serif', size=14)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)
cmap = plt.get_cmap('magma') # # tried: 'RdBu', plasma, inferno
plt.rc('image', cmap=cmap.name)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'

# It would be nice to draw an arrow to the point where the load is applied.

def plot_field(
        coords: dict[str, np.ndarray],
        truth_field: np.ndarray, # (n_x, n_y, n_z, 3)
        pred_field: np.ndarray,# (n_x, n_y, n_z, 3)
        input_function_labels: list[str],
        input_function_values: np.ndarray,
        target_labels: list[str],
        plot_plane: str,
        plotted_variable: str
    ) -> plt.Figure:

    coords_keys = ['x', 'y', 'z']

    plotted_label_map = {
        'predictions': '$\\mathbf{u}_{pred}$',
        'truths': '$\\mathbf{u}_{truth}$',
        'rel_errors': 'Relative error',
    }

    dims = tuple(i for i in coords_keys if i in plot_plane)
    orthogonal_dim = [pos for pos, coord in enumerate(coords_keys) if coord not in plot_plane][0]
    dim1, dim2 = dims

    coord1 = coords[dim1]
    coord2 = coords[dim2]

    if np.all(coord1 >= 0):
        coord1_full = np.concatenate((-np.flip(coord1), coord1))
    else:
        coord1_full = coord1

    X, Y = np.meshgrid(coord1_full, coord2, indexing='ij')

    param_str = format_param(
        param=input_function_values,
        param_keys=input_function_labels
    )

    x_label, y_label = f'${dim1}$', f'${dim2}$'

    ncols = 3

    mask = [np.s_[:]] * pred_field.ndim
    axis_length = pred_field.shape[orthogonal_dim + 1]
    middle_index =  axis_length // 2 if axis_length % 2 != 0 else axis_length // 2 + 1
    
    mask[orthogonal_dim + 1] = middle_index


    plot_pred = pred_field[tuple(mask)]
    plot_truth = truth_field[tuple(mask)]


    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(
        5 * ncols, 4), sharex=True, sharey=True)
    
    x_center = (X.min() + X.max()) / 2
    y_center = (Y.min() + Y.max()) / 2

    annotation_color = 'black'
    arrow_props = dict(
        arrowstyle="->",  # Or "->", "fancy", etc. for different arrow styles
        color=annotation_color,          # Color of the arrow
        lw=0.9,                 
        mutation_scale=20,    
        shrinkA=0,            
        shrinkB=0             
    )

    arrow_length = (Y.max() - Y.min()) * 0.25

    for ax, pred, truth, ch in zip(axes, plot_pred, plot_truth, target_labels):

        rel_error = (pred - truth) / truth

        plotted = pred if plotted_variable == 'predictions' else \
            truth if plotted_variable == 'truths' else rel_error
        
        c = ax.contourf(X, Y, plotted, cmap=cmap.name)
        ax.set_title(f"{ch} {param_str}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(c, ax=ax)

        if plot_plane != 'xy':
            ax.annotate(
                '$\\mathbf{P}$',                                  
                xy=(x_center, y_center), 
                xytext=(x_center, y_center + arrow_length),         
                arrowprops=arrow_props,
                color=annotation_color,
                ha='center',                         
                va='center'                          
            )
    fig.suptitle(f"{plotted_label_map[plotted_variable]} ${plot_plane}$ - Plane")
        
    fig.tight_layout()
    return fig
