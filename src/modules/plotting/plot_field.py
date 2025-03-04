import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..pipe.preprocessing import format_param

def plot_2D_field(coords, truth_field=None, pred_field=None, param_value=None, param_labels=None):
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
        pred_field (ndarray, optional): Predicted field in flattened trunk format.
        param_value: A parameter value (or tuple/dict) to display in the title.
        coord_labels (dict, optional): Dictionary with keys:
            - 'plot_dims': tuple of two strings indicating which coordinate keys to plot, e.g. ('x','z')
            - For each coordinate key, a label string.
            If not provided, defaults to the first two keys in coords.
    
    Returns:
        fig: A matplotlib Figure object.
    """
    # Determine plotting dimensions.
    dims = list(coords.keys())
    if len(dims) < 2:
        raise ValueError("At least two coordinate arrays are required for plotting.")
    dim1, dim2 = dims[:2]
    
    # Retrieve coordinate arrays.
    coord1 = coords[dim1]
    coord2 = coords[dim2]
    
    # Mirror the horizontal axis (assumed to be coord1) if all values are positive.
    if np.all(coord1 >= 0):
        # Mirror by concatenating the negative of coord1 (flipped, excluding the first element) with coord1.
        coord1_full = np.concatenate((-np.flip(coord1[1:]), coord1))
    else:
        coord1_full = coord1

    # Check for extra coordinate arrays.
    remaining_keys = [k for k in coords if k not in (dim1, dim2)]
    if remaining_keys:
        rem_key = remaining_keys[0]
        rem_array = coords[rem_key]
        slice_index = len(rem_array) // 2
        n1 = len(coord1)
        n2 = len(coord2)
        n_rem = len(rem_array)
        if truth_field is not None:
            truth_mesh = truth_field.reshape(n1, n2, n_rem)[:, :, slice_index]
        if pred_field is not None:
            pred_mesh = pred_field.reshape(n1, n2, n_rem)[:, :, slice_index]
    else:
        n1 = len(coord1)
        n2 = len(coord2)
        if truth_field is not None:
            truth_mesh = truth_field.reshape(n1, n2)
        if pred_field is not None:
            pred_mesh = pred_field.reshape(n1, n2)
    
    # If mirroring is applied, mirror the field along the horizontal axis.
    if np.all(coord1 >= 0):
        if truth_field is not None:
            truth_mesh = np.concatenate((np.flip(truth_mesh[1:], axis=0), truth_mesh), axis=0)
        if pred_field is not None:
            pred_mesh = np.concatenate((np.flip(pred_mesh[1:], axis=0), pred_mesh), axis=0)
        # Update n1 to the length of the mirrored coordinate.
        n1 = len(coord1_full)
    
    # Create a 2D meshgrid using the full horizontal coordinate.
    X, Y = np.meshgrid(coord1_full, coord2, indexing='ij')
    
    # Format parameter value.
    param_str = format_param(param_value, param_keys=param_labels)
    
    # Determine axis labels.
    x_label, y_label = dim1, dim2

    # Check if we are dealing with complex data.
    is_complex = False
    if truth_field is not None and np.iscomplexobj(truth_field):
        is_complex = True
    if pred_field is not None and np.iscomplexobj(pred_field):
        is_complex = True

    if is_complex:
        # For complex fields, split into real and imaginary parts.
        if truth_field is not None:
            truth_real = truth_mesh.real
            truth_imag = truth_mesh.imag
        if pred_field is not None:
            pred_real = pred_mesh.real
            pred_imag = pred_mesh.imag
        
        # If both fields are provided, plot prediction, label, and error; otherwise, plot one column.
        ncols = 3 if (truth_field is not None and pred_field is not None) else 1
        fig, ax = plt.subplots(2, ncols, figsize=(6 * ncols, 10), sharex=True, sharey=True)
        
        # Row 0: Real parts.
        if ncols == 3:
            norm_real = colors.Normalize(vmin=min(np.min(pred_real), np.min(truth_real)),
                                         vmax=max(np.max(pred_real), np.max(truth_real)))
            c0 = ax[0, 0].contourf(X, Y, pred_real, cmap='viridis', norm=norm_real)
            ax[0, 0].set_title(f"Real Pred {param_str}")
            ax[0, 0].set_xlabel(x_label)
            ax[0, 0].set_ylabel(y_label)
            
            c1 = ax[0, 1].contourf(X, Y, truth_real, cmap='viridis', norm=norm_real)
            ax[0, 1].set_title(f"Real Label {param_str}")
            ax[0, 1].set_xlabel(x_label)
            
            c2 = ax[0, 2].contourf(X, Y, np.abs(truth_real - pred_real), cmap='viridis')
            ax[0, 2].set_title(f"Real Abs Error {param_str}")
            ax[0, 2].set_xlabel(x_label)
            ax[0 , 2].invert_yaxis()

            fig.colorbar(c0, ax=ax[0, 0])
            fig.colorbar(c1, ax=ax[0, 1])
            fig.colorbar(c2, ax=ax[0, 2])
        else:
            field = pred_real if pred_field is not None else truth_real
            c0 = ax[0, 0].contourf(X, Y, field, cmap='viridis')
            ax[0, 0].set_title(f"Real Field {param_str}")
            ax[0, 0].set_xlabel(x_label)
            ax[0, 0].set_ylabel(y_label)
            fig.colorbar(c0, ax=ax[0, 0])

        # Row 1: Imaginary parts.
        if ncols == 3:
            norm_imag = colors.Normalize(vmin=min(np.min(pred_imag), np.min(truth_imag)),
                                         vmax=max(np.max(pred_imag), np.max(truth_imag)))
            c3 = ax[1, 0].contourf(X, Y, pred_imag, cmap='viridis', norm=norm_imag)
            ax[1, 0].set_title(f"Imag Pred {param_str}")
            ax[1, 0].set_xlabel(x_label)
            ax[1 , 0].invert_yaxis()
            ax[1, 0].set_ylabel(y_label)
            
            c4 = ax[1, 1].contourf(X, Y, truth_imag, cmap='viridis', norm=norm_imag)
            ax[1, 1].set_title(f"Imag Label {param_str}")
            ax[1, 1].set_xlabel(x_label)
            ax[1 , 1].invert_yaxis()
            
            c5 = ax[1, 2].contourf(X, Y, np.abs(truth_imag - pred_imag), cmap='viridis')
            ax[1, 2].set_title(f"Imag Abs Error {param_str}")
            ax[1, 2].set_xlabel(x_label)
            ax[1 , 2].invert_yaxis()

            fig.colorbar(c3, ax=ax[1, 0])
            fig.colorbar(c4, ax=ax[1, 1])
            fig.colorbar(c5, ax=ax[1, 2])
        else:
            field = pred_imag if pred_field is not None else truth_imag
            c3 = ax[1, 0].contourf(X, Y, field, cmap='viridis')
            ax[1, 0].set_title(f"Imag Field {param_str}")
            ax[1, 0].set_xlabel(x_label)
            ax[1 , 0].invert_yaxis()
            ax[1, 0].set_ylabel(y_label)

            fig.colorbar(c3, ax=ax[1, 0])
        
        # Invert vertical axis for all subplots.
        # for a in np.concatenate((ax[0, :], ax[1, :])):
        #     a.invert_yaxis()
        ax[0, 0].invert_yaxis()
        fig.tight_layout()
    else:
        # For real fields.
        norm = colors.Normalize(vmin=min(np.min(truth_mesh), np.min(pred_mesh)),
                                  vmax=max(np.max(truth_mesh), np.max(pred_mesh)))
        error_mesh = np.abs(truth_mesh - pred_mesh)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        c0 = ax[0].contourf(X, Y, pred_mesh, cmap='viridis', norm=norm)
        ax[0].set_title(f"Prediction {param_str}")
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel(y_label)
        
        c1 = ax[1].contourf(X, Y, truth_mesh, cmap='viridis', norm=norm)
        ax[1].set_title(f"Label {param_str}")
        ax[1].set_xlabel(x_label)
        
        c2 = ax[2].contourf(X, Y, error_mesh, cmap='viridis')
        ax[2].set_title(f"Absolute Error {param_str}")
        ax[2].set_xlabel(x_label)
        
        for a in ax:
            a.invert_yaxis()
        fig.colorbar(c0, ax=ax[0])
        fig.colorbar(c1, ax=ax[1])
        fig.colorbar(c2, ax=ax[2])
        fig.tight_layout()
    return fig
