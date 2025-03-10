import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
from matplotlib import _pylab_helpers
import matplotlib.ticker as ticker

def plot_coefficients(modes: np.ndarray, coefficients_mean_abs: np.ndarray, modes_to_highlight: np.ndarray[int], **kwargs) -> plt.figure:
    
    # modes are: (n_modes, coord1, coord1, n_channels_basis)
    # coeffs_mean_abs are: (n_modes, n_channels_coeffs)
    # modes to highlight are: (k-highest, n_channels_basis)
    # Function will show the k modes with the highest
    output_labels = kwargs.get('label_mapping')
    n_channels_basis = modes.shape[-1]
    n_channels_coeffs = coefficients_mean_abs.shape[-1]
    k_highest_modes = len(modes_to_highlight)
    fig, ax = plt.subplots(ncols=n_channels_coeffs, figsize=(8 * n_channels_coeffs, 7))
    for coeff_channel in range(n_channels_coeffs):
        coefficients_mean_abs_for_i_channel = coefficients_mean_abs[..., coeff_channel]
        x = np.arange(len(coefficients_mean_abs_for_i_channel))
        ax[coeff_channel].bar(x, coefficients_mean_abs_for_i_channel, width=0.8, color='steelblue', alpha=1) # should be n_channel columns
        ax[coeff_channel].set_yscale('log')
        ax[coeff_channel].set_ylim([None, 100 * coefficients_mean_abs_for_i_channel.max()])
        ax[coeff_channel].tick_params(axis='y', labelsize=12)
        ax[coeff_channel].set_xlabel('Modes', fontsize=14)
        ax[coeff_channel].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x + 1)}"))
        if output_labels:
            if isinstance(output_labels[coeff_channel], str):
                output_name = output_labels[coeff_channel]
            elif isinstance(output_labels[coeff_channel], dict):
                output_name = (output_labels[coeff_channel].values()[0])
            else:
                raise ValueError(f"Legend with label type: {type(output_labels[coeff_channel])} not implemented.")
            ax[coeff_channel].set_ylabel(f'Branch coefficient mean ({output_name})', fontsize=14)
        else:
            ax[coeff_channel].set_ylabel(f'Branch coefficient mean for output {coeff_channel + 1}', fontsize=14)

        positions = []
        for i in range(k_highest_modes):
            x_pos = 0.1 + (i / k_highest_modes) * 0.8
            y_pos = 0.75 - (i % 2) * 0.25
            positions.append((x_pos, y_pos))

        for (index, pos) in zip(modes_to_highlight[..., coeff_channel], positions):
            ax_inset = ax[coeff_channel].inset_axes([pos[0], pos[1], 0.16, 0.2])
            mode_reshaped = np.concatenate((np.flip(modes[index][1:], axis=0), modes[index]), axis=0)
            if n_channels_basis > 1:
                mode_reshaped_channel = mode_reshaped[..., coeff_channel]
            else:
                mode_reshaped_channel= mode_reshaped[..., 0]
            mode_data = np.flipud(np.transpose(mode_reshaped_channel, (1, 0)))
            ax_inset.imshow(mode_data, origin='lower')
            ax_inset.set_title(f'Mode={index + 1}', fontsize=12)
            ax_inset.set_xticklabels([])
            ax_inset.set_yticklabels([])
            ax_inset.tick_params(labelsize=7)

            x_main = index
            y_main = coefficients_mean_abs[index, coeff_channel]
            coordsA = 'data'
            coordsB = 'axes fraction'
            for corner in [(0.5, 0)]:
                con = ConnectionPatch(
                    xyA=(x_main, y_main), coordsA=coordsA,
                    xyB=corner, coordsB=coordsB,
                    axesA=ax[coeff_channel], axesB=ax_inset,
                    color='black'
                )

                ax_inset.add_artist(con)

    # fig.tight_layout()
    return fig