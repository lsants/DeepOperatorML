import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import format_param

def plot_axis(coords, truth_field, pred_field, param_value, axis=None,
                        non_dim_plot=True, rotated=False, coord_labels=None):
    """
    Plots line comparisons along a chosen axis (or along two axes if none is specified)
    for the predicted and true fields. This function is modular: it accepts a dictionary of 
    coordinate arrays and optional coordinate labels, and can handle complex outputs by plotting 
    real and imaginary parts separately.
    
    Args:
        coords (dict): Dictionary of coordinate arrays. For example, for a Cartesian system,
            e.g. {'x': x_array, 'y': y_array, 'z': z_array}.
        truth_field (ndarray): Ground truth field in flattened trunk format.
        pred_field (ndarray): Predicted field in flattened trunk format.
        param_value: Parameter value(s) to display in the title.
        axis (str or None): If provided, must be one of the keys in coords; the function will extract
            a line slice along this coordinate (by fixing it at its minimum value). If None, the function
            will produce plots along the first two coordinate dimensions.
        non_dim_plot (bool): Whether to use non-dimensional labels.
        rotated (bool): If True, swap axes in the line plot (useful for certain cases).
        coord_labels (dict, optional): Dictionary that may contain:
            - 'plot_dims': tuple of two strings indicating which coordinate keys to use for plotting (if axis is None),
              e.g. ('x','y').
            - For each coordinate key, a label string (e.g. {'x': 'x (m)', 'y': 'y (m)', 'z': 'z (m)'}).
    
    Returns:
        fig: A matplotlib Figure object.
    """
    # Format the parameter value.
    param_str = format_param(param_value)
    
    # Determine axis labels for the coordinates.
    if coord_labels is not None:
        # If plot_dims is provided and axis is None, use those.
        if axis is None and 'plot_dims' in coord_labels:
            dim1, dim2 = coord_labels['plot_dims']
        else:
            # Otherwise, use axis if provided, and choose one other dimension arbitrarily.
            if axis is not None and axis in coords:
                dim_fixed = axis
                other_dims = [k for k in coords.keys() if k != axis]
                # Pick the first available dimension as the varying one.
                dim_vary = other_dims[0] if other_dims else None
            else:
                dims = list(coords.keys())
                if len(dims) < 2:
                    raise ValueError("At least two coordinate arrays are required for axis plotting.")
                dim1, dim2 = dims[:2]
    else:
        dims = list(coords.keys())
        if len(dims) < 2:
            raise ValueError("At least two coordinate arrays are required for axis plotting.")
        if axis is None:
            dim1, dim2 = dims[:2]
        else:
            if axis not in dims:
                raise ValueError(f"Axis '{axis}' not found in coordinate keys.")
            dim_fixed = axis
            other_dims = [k for k in dims if k != axis]
            dim_vary = other_dims[0]
    
    # If axis is provided, we extract a 1D line by fixing the chosen coordinate at its minimum.
    if axis is not None:
        fixed_coord = coords[dim_fixed]
        vary_coord = coords[dim_vary]
        # For the fixed coordinate, select the minimum value (or you could choose the middle value).
        fixed_value = np.min(fixed_coord)
        # Find the index in the fixed coordinate array that is closest to fixed_value.
        fixed_idx = np.argmin(np.abs(fixed_coord - fixed_value))
        # Reshape truth_field and pred_field into a meshgrid corresponding to the full coordinates.
        # We assume that the flattened field has shape (n1 * n2 * ...), where the order is given by
        # the order of coords keys as in the meshgrid created with 'ij' indexing.
        # Here, we need to determine the shape from the coordinate arrays.
        grid_shape = tuple(len(coords[k]) for k in coords)
        # For simplicity, we assume that the order of dimensions in the meshgrid is the same as the
        # sorted order of the keys in coords.
        sorted_keys = sorted(coords.keys())
        field_reshaped = truth_field.reshape(*[len(coords[k]) for k in sorted_keys])
        pred_reshaped = pred_field.reshape(*[len(coords[k]) for k in sorted_keys])
        # Now, to extract a line along dimension dim_vary, we fix the coordinate corresponding to dim_fixed.
        # We need to find the index of dim_fixed in sorted_keys.
        fixed_idx_in_order = sorted_keys.index(dim_fixed)
        vary_idx_in_order = sorted_keys.index(dim_vary)
        # Now, extract a slice: fix the fixed dimension at fixed_idx_in_order, and flatten the remaining.
        # We assume the line is along the varying dimension.
        # First, bring the varying dimension to the front.
        truth_line = np.take(field_reshaped, indices=fixed_idx, axis=fixed_idx_in_order)
        pred_line = np.take(pred_reshaped, indices=fixed_idx, axis=fixed_idx_in_order)
        var = vary_coord
        # Set labels.
        x_label = coord_labels.get(dim_vary, dim_vary) if coord_labels else dim_vary
        y_label = coord_labels.get(dim_fixed, dim_fixed) if coord_labels else dim_fixed
        # Plot the line. We handle complex fields below.
        is_complex = np.iscomplexobj(truth_line) or np.iscomplexobj(pred_line)
        if is_complex:
            truth_real = truth_line.real
            truth_imag = truth_line.imag
            pred_real = pred_line.real
            pred_imag = pred_line.imag
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].plot(var, truth_real, '.-k', label='Label (real)')
            ax[0].plot(var, pred_real, 'xr', label='Pred (real)')
            ax[0].set_xlabel(x_label)
            ax[0].set_ylabel("Real part")
            ax[0].legend()
            ax[1].plot(var, truth_imag, '.-k', label='Label (imag)')
            ax[1].plot(var, pred_imag, 'xr', label='Pred (imag)')
            ax[1].set_xlabel(x_label)
            ax[1].set_ylabel("Imaginary part")
            ax[1].legend()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(var, truth_line, '.-k', label='Label')
            ax[0].set_xlabel(x_label)
            ax[0].set_ylabel("Field")
            ax[0].set_title(f"Label (param: {format_param(param_value)})")
            ax[0].legend()
            ax[1].plot(var, pred_line, 'xr', label='Prediction')
            ax[1].set_xlabel(x_label)
            ax[1].set_ylabel("Field")
            ax[1].set_title(f"Prediction (param: {format_param(param_value)})")
            ax[1].legend()
    else:
        # If no specific axis is provided, plot line comparisons for both of the first two dimensions.
        # Here we extract two slices: one along the first dimension (fix the second at its minimum)
        # and one along the second dimension (fix the first at its minimum).
        sorted_keys = sorted(coords.keys())
        dim1, dim2 = sorted_keys[:2]
        coord1 = coords[dim1]
        coord2 = coords[dim2]
        
        # Reshape the flattened fields to a meshgrid format.
        shape_2d = (len(coord1), len(coord2))
        truth_mesh = truth_field.reshape(*shape_2d)
        pred_mesh  = pred_field.reshape(*shape_2d)
        
        # For the dim1 profile, fix dim2 at its minimum.
        slice_idx_dim2 = np.argmin(coord2)
        line_dim1_truth = truth_mesh[:, slice_idx_dim2]
        line_dim1_pred  = pred_mesh[:, slice_idx_dim2]
        # For the dim2 profile, fix dim1 at its minimum.
        slice_idx_dim1 = np.argmin(coord1)
        line_dim2_truth = truth_mesh[slice_idx_dim1, :]
        line_dim2_pred  = pred_mesh[slice_idx_dim1, :]
        
        # Set up labels.
        x_label_dim1 = coord_labels.get(dim1, dim1) if coord_labels else dim1
        x_label_dim2 = coord_labels.get(dim2, dim2) if coord_labels else dim2
        
        is_complex = np.iscomplexobj(truth_field) or np.iscomplexobj(pred_field)
        fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        if is_complex:
            # For dim1 profile, plot real and imaginary parts.
            ax[0, 0].plot(coord1, line_dim1_truth.real, '.-k', label='Label (real)')
            ax[0, 0].plot(coord1, line_dim1_pred.real, 'xr', label='Pred (real)')
            ax[0, 0].set_xlabel(x_label_dim1)
            ax[0, 0].set_ylabel(f"{dim1} (real)")
            ax[0, 0].legend()
            
            ax[0, 1].plot(coord1, line_dim1_truth.imag, '.-k', label='Label (imag)')
            ax[0, 1].plot(coord1, line_dim1_pred.imag, 'xr', label='Pred (imag)')
            ax[0, 1].set_xlabel(x_label_dim1)
            ax[0, 1].set_ylabel(f"{dim1} (imag)")
            ax[0, 1].legend()
            
            ax[0, 2].plot(coord1, np.abs(line_dim1_truth - line_dim1_pred), '.-k', label='Abs Error')
            ax[0, 2].set_xlabel(x_label_dim1)
            ax[0, 2].set_ylabel(f"{dim1} Error")
            ax[0, 2].legend()
            
            # For dim2 profile.
            ax[1, 0].plot(coord2, line_dim2_truth.real, '.-k', label='Label (real)')
            ax[1, 0].plot(coord2, line_dim2_pred.real, 'xr', label='Pred (real)')
            ax[1, 0].set_xlabel(x_label_dim2)
            ax[1, 0].set_ylabel(f"{dim2} (real)")
            ax[1, 0].legend()
            
            ax[1, 1].plot(coord2, line_dim2_truth.imag, '.-k', label='Label (imag)')
            ax[1, 1].plot(coord2, line_dim2_pred.imag, 'xr', label='Pred (imag)')
            ax[1, 1].set_xlabel(x_label_dim2)
            ax[1, 1].set_ylabel(f"{dim2} (imag)")
            ax[1, 1].legend()
            
            ax[1, 2].plot(coord2, np.abs(line_dim2_truth - line_dim2_pred), '.-k', label='Abs Error')
            ax[1, 2].set_xlabel(x_label_dim2)
            ax[1, 2].set_ylabel(f"{dim2} Error")
            ax[1, 2].legend()
        else:
            # For real fields.
            ax[0, 0].plot(coord1, line_dim1_truth, '.-k', label='Label')
            ax[0, 0].plot(coord1, line_dim1_pred, 'xr', label='Pred')
            ax[0, 0].set_xlabel(x_label_dim1)
            ax[0, 0].set_ylabel(dim1)
            ax[0, 0].legend()
            
            ax[0, 1].plot(coord1, np.abs(line_dim1_truth - line_dim1_pred), '.-k', label='Abs Error')
            ax[0, 1].set_xlabel(x_label_dim1)
            ax[0, 1].set_ylabel(f"{dim1} Error")
            ax[0, 1].legend()
            
            ax[0, 2].axis('off')
            
            ax[1, 0].plot(coord2, line_dim2_truth, '.-k', label='Label')
            ax[1, 0].plot(coord2, line_dim2_pred, 'xr', label='Pred')
            ax[1, 0].set_xlabel(x_label_dim2)
            ax[1, 0].set_ylabel(dim2)
            ax[1, 0].legend()
            
            ax[1, 1].plot(coord2, np.abs(line_dim2_truth - line_dim2_pred), '.-k', label='Abs Error')
            ax[1, 1].set_xlabel(x_label_dim2)
            ax[1, 1].set_ylabel(f"{dim2} Error")
            ax[1, 1].legend()
            
            ax[1, 2].axis('off')
        fig.suptitle(f"Axis Comparison (param: {format_param(param_value)})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig