import numpy as np
import matplotlib.pyplot as plt

def plot_basis_function(coords, basis, strategy, **kwargs):
    """
    Plots a single basis function on a 2D plane using the provided coordinate system.
    
    The function assumes that the input `basis` is already in the correct shape:
       (n_coord1, n_coord2, ..., n_channels)
    where the first two dimensions correspond to the two plotting dimensions and the last 
    dimension contains the channel(s) (e.g. for real and/or imaginary parts).
    
    If there is more than one channel, they are plotted side-by-side in one row.
    
    Additional keyword arguments:
      - full (bool, optional): Whether to mirror the horizontal axis if all its values are positive.
                                 Default is True.
      - coord_labels (dict, optional): Dictionary with optional keys:
             * 'plot_dims': tuple of two strings indicating which coordinate keys to plot,
               e.g. ('x','z'). If not provided, the first two keys in `coords` are used.
             * For each coordinate key, a label string (e.g. {'x': 'x (m)', 'z': 'z (m)'}).
      - param_value: A parameter value to include in the title.
    
    Args:
      coords (dict): Dictionary where keys are coordinate names (e.g. 'x', 'y', 'z')
                     and values are 1D numpy arrays.
      basis (ndarray): A single basis function array of shape (n_coord1, n_coord2, n_channels).
      strategy (str): A string describing the method used to compute the basis.
      **kwargs: See above.
    
    Returns:
      fig: A matplotlib Figure object.
    """

    full = kwargs.get('full', True)
    coord_labels = kwargs.get('coord_labels')
    output_keys = kwargs.get('output_keys')
    index = kwargs.get('index')
    if coord_labels is not None and 'plot_dims' in coord_labels:
        plot_dim1, plot_dim2 = coord_labels['plot_dims']
    else:
        dims = list(coords.keys())
        if len(dims) < 2:
            raise ValueError("At least two coordinate arrays are required for plotting 2D basis functions.")
        plot_dim1, plot_dim2 = dims[:2]
    
    horiz = coords[plot_dim1]
    vert = coords[plot_dim2]
    
    if full and np.all(horiz >= 0):
        horiz_full = np.concatenate((-np.flip(horiz[1:]), horiz))
    else:
        horiz_full = horiz
    n_h_full = len(horiz_full)
    
    n1, n2, n_channels = basis.shape
    if n1 != len(horiz) or n2 != len(vert):
        raise ValueError(f"Basis grid shape ({n1}, {n2}) does not match coordinate lengths "
                         f"({len(horiz)}, {len(vert)})")
    
    if full and np.all(horiz >= 0):
        basis = np.concatenate((np.flip(basis[1:], axis=0), basis), axis=0)
    
    X, Y = np.meshgrid(horiz_full, vert, indexing='ij')
    if X.shape != (n_h_full, len(vert)):
        raise ValueError(f"Meshgrid shape mismatch: X.shape={X.shape}, expected {(n_h_full, len(vert))}")
    
    
    # Set axis labels.
    if coord_labels is not None:
        x_label = coord_labels.get(plot_dim1, plot_dim1)
        y_label = coord_labels.get(plot_dim2, plot_dim2)
    else:
        x_label, y_label = plot_dim1, plot_dim2
    
    title = f"Basis Functions ({strategy})"
    if n_channels == 1:
        label_mapping = ''
    else:
        label_mapping = kwargs.get('label_mapping')
    
    # Plot each channel in the basis function on the same row.
    ncols = n_channels
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    axs = axs.flatten()
    if output_keys:
        output_map = {ch : key for ch, key in zip(range(n_channels), output_keys)}
    for ch in range(n_channels):
        # Extract channel ch.
        field_ch = basis[:, :, ch]
        # Check shape: should be (n_h_full, len(vert))
        if field_ch.shape != X.shape:
            if field_ch.T.shape == X.shape:
                field_ch = field_ch.T
            else:
                raise ValueError(f"Shape mismatch in channel {output_map[ch]}: field shape {field_ch.shape} vs X shape {X.shape}")
        contour = axs[ch].contourf(X, Y, field_ch, cmap="viridis")
        axs[ch].invert_yaxis()
        axs[ch].set_xlabel(x_label, fontsize=12)
        axs[ch].set_ylabel(y_label, fontsize=12)
        if label_mapping is not None and label_mapping != '':
            axs[ch].set_title(f"Channel {label_mapping[ch]}, vector {index}", fontsize=12)
        elif label_mapping is not None:
            axs[ch].set_title(f"Vector {index}", fontsize=12)
        else:
            axs[ch].set_title(f"Channel {ch+1}, vector {index}", fontsize=12)

        fig.colorbar(contour, ax=axs[ch])
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig
