import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)
cmap = plt.get_cmap('magma') # tried: 'RdBu'
plt.rc('image', cmap=cmap.name)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_basis(
        coords: dict[str, np.ndarray],
        basis: np.ndarray,
        index: int,
        target_labels: list[str],
        plot_plane: str,
    ) -> plt.Figure:

    coords_keys = ['x', 'y', 'z']
    orthogonal_dim = [pos for pos, coord in enumerate(coords_keys) if coord not in plot_plane][0]
    plot_dim1, plot_dim2 = [c for c in coords_keys if c in plot_plane]

    horiz = coords[plot_dim1]
    vert = coords[plot_dim2]

    n_channels, n1, n2, n2 = basis.shape

    X, Y = np.meshgrid(horiz, vert, indexing='ij')

    x_label, y_label = f'${plot_dim1}$', f'${plot_dim2}$'

    ncols = n_channels
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    axs = axs.flatten()


    mask = [np.s_[:]] * basis.ndim
    axis_length = basis.shape[orthogonal_dim + 1]
    middle_index =  axis_length // 2 if axis_length % 2 != 0 else axis_length // 2 + 1
    
    mask[orthogonal_dim + 1] = middle_index

    basis_2d = basis[tuple(mask)]

    for ch, ax in enumerate(axs):
        suffix = 'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'
        sup_title = f"{index}{suffix} vector"
        sub_title = f"{index}{suffix} vector for {target_labels[ch]}"

        c = ax.contourf(X, Y, basis_2d[ch], cmap=cmap.name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(c, ax=ax)
        if n_channels > 1:
            ax.set_title(sub_title)

    if n_channels == 1:
        fig.suptitle(f"{sup_title} ${plot_plane}$ - Plane")
        
    fig.tight_layout()

    return fig
