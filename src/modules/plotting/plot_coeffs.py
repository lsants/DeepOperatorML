import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch

def plot_coefficients(modes: np.ndarray, coefficients: np.ndarray, modes_to_highlight: list[int]) -> plt.figure:
    
    from matplotlib import _pylab_helpers

    # modes are: (n_modes, coord1, coord1, channels)
    # coeffs are: (input_functions, n_modes)
    # modes to highlight are [i, j, k, l]
    branch_coeff_mean = np.mean(abs(coefficients), axis=0)

    fig, ax = plt.subplots(figsize=(8,8))

    # Plot the main bar graph on a log scale
    ax.bar(np.arange(len(branch_coeff_mean)), branch_coeff_mean, width=0.6, color='steelblue', alpha=1)
    ax.set_yscale('log')
    ax.set_ylim([0, 20])

    ax.set_xlabel('Modes')
    ax.set_ylabel('Branch coefficient mean')
    ax.set_title('Modes vs. Branch Coefficient mean'
    )

    positions = [(0.1, 0.7), (0.3, 0.7), (0.5, 0.7), (0.67, 0.7), (0.83, 0.7)]  # Adjust coordinates as needed

    for index, pos in zip(modes_to_highlight, positions):
        ax_inset = ax.inset_axes([pos[0], pos[1], 0.15, 0.35])
        mode_reshaped = np.concatenate((np.flip(modes[index][1:], axis=0), modes[index]), axis=0)
        mode_data = np.flipud(mode_reshaped.T) # placeholder data
        ax_inset.imshow(mode_data, origin='lower')
        ax_inset.set_title(f'Mode={index}', fontsize=8)
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.tick_params(labelsize=6)

        x_main = index
        y_main = branch_coeff_mean[index]
        coordsA = 'data'
        coordsB = 'axes fraction'
        for corner in [(0, 0.5)]:
            con = ConnectionPatch(
                xyA=(x_main, y_main), coordsA=coordsA,
                xyB=corner, coordsB=coordsB,
                axesA=ax, axesB=ax_inset,
                color='black'
            )
            ax_inset.add_artist(con)

    fig.tight_layout()
    return fig