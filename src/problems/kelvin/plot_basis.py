import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)
cmap = plt.get_cmap('viridis') # tried: 'RdBu'
plt.rc('image', cmap=cmap.name)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_basis(coords: dict[str, np.ndarray],
               basis: np.ndarray,
               index: int,
               target_labels: list[str]) -> plt.Figure:
    label_mapping = target_labels
    plot_dim1, plot_dim2 = [c for c in coords]

    horiz = coords[plot_dim1]
    vert = coords[plot_dim2]

    horiz_full = np.concatenate((-np.flip(horiz), horiz))
    n_h_full = len(horiz_full)

    n_channels, n1, n2 = basis.shape
    if n1 != len(horiz_full) or n2 != len(vert):
        raise ValueError(f"Basis grid shape ({n1}, {n2}) does not match coordinate lengths "
                         f"({len(horiz_full)}, {len(vert)})")

    X, Y = np.meshgrid(horiz_full, vert, indexing='ij')
    if X.shape != (n_h_full, len(vert)):
        raise ValueError(
            f"Meshgrid shape mismatch: X.shape={X.shape}, expected {(n_h_full, len(vert))}")

    x_label, y_label = plot_dim1, plot_dim2

    ncols = n_channels
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    axs = axs.flatten()
    if target_labels:
        output_map = {ch: key for ch, key in zip(
            range(n_channels), target_labels)}

    for ch in range(n_channels):
        suffix = 'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'
        sup_title = f"{index}{suffix} vector"
        sub_title = f"{index}{suffix} vector for {label_mapping[ch]}"
        field_ch = basis[ch]
        if field_ch.shape != X.shape:
            raise ValueError(
                f"Shape mismatch in channel {output_map[ch]}: field shape {field_ch.shape} vs X shape {X.shape}")

        contour = axs[ch].contourf(X, Y, field_ch)
        # contour = axs[ch].imshow(np.flipud(field_ch.T))
        axs[ch].invert_yaxis()
        axs[ch].set_xlabel(x_label, fontsize=12)
        axs[ch].set_ylabel(y_label, fontsize=12)
        if n_channels > 1:
            axs[ch].set_title(sub_title)

        fig.colorbar(contour, ax=axs[ch])
    if n_channels == 1:
        fig.suptitle(sup_title)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
