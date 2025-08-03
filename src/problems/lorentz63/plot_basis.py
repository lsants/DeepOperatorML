import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_basis_component(
        coords: dict[str, np.ndarray],
        basis: np.ndarray,
        index: int,
        target_labels: list[str],
    ) -> plt.Figure:


    n_channels, n_t = basis.shape

    x_label = 't'
    y_labels = ['$\\mathbf{r}(t)}$'] if n_channels == 1 else target_labels

    nrows = n_channels
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(5, 3 * nrows))
    colors = ['cyan', 'magenta', 'goldenrod']
    for ch, ax in enumerate(axs):
        suffix = 'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'
        sup_title = f"{index}{suffix} vector"
        sub_title = f"{index}{suffix} vector for {target_labels[ch]}"
        ax.plot(coords['t'], basis[ch], lw=1.2, color=colors[ch])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[ch])
        if n_channels > 1:
            ax.set_title(sub_title)

    if n_channels == 1:
        fig.suptitle(f"{sup_title}")
        
    fig.tight_layout()

    return fig

def plot_basis_3d(
        coords: dict[str, np.ndarray],
        basis: np.ndarray,
        index: int,
        target_labels: list[str],
    ) -> plt.Figure:

    x_label = 't'
    y_labels = '$\\mathbf{r}(t)$'
    suffix = 'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'

    fig = plt.figure("3D Basis")
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = basis
    
    ax.plot(x, y, z, lw=1.2, label=f"{index}{suffix} vector", color='blue')
    sub_title = f"{index}{suffix} vector"
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z')
    ax.set_title(sub_title)

    fig.tight_layout()

    return fig
