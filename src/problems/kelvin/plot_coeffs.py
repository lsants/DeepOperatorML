import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker

plt.rc('font', family='serif', size=14)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_coefficients_mean(
    vectors: np.ndarray,
    coefficients: np.ndarray,
    num_vectors_to_highlight: int,
    target_labels: list[str]
) -> plt.Figure:

    n_basis = vectors.shape[1]
    coefficients_mean = coefficients.mean(axis=0).T
    n_channels = coefficients_mean.shape[-1]
    k_highest_modes = np.sort(np.argsort(-np.abs(coefficients_mean), axis=0)[
        : num_vectors_to_highlight, :], axis=0)  # [k, n_basis]

    fig, ax = plt.subplots(ncols=n_channels, figsize=(5 * n_channels, 5))

    for ch in range(n_channels):
        coefficients_mean_for_i_channel = coefficients_mean[..., ch]
        x = np.arange(len(coefficients_mean_for_i_channel))
        ax[ch].bar(x, abs(coefficients_mean_for_i_channel), align='edge',
                   color='steelblue', alpha=0.8)
        ax[ch].set_ylim([None, 1.3*abs(coefficients_mean_for_i_channel).max()])
        ax[ch].tick_params(axis='y', labelsize=12)
        ax[ch].set_xlabel(r'$i$', fontsize=15)
        ax[ch].xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))

        output_name = target_labels[ch]
        ax[ch].set_ylabel(
            r'$i$' + f'-th Coefficient mean', fontsize=15)
        ax[ch].set_title(f'{output_name}')
        positions = []
        for i in range(len(k_highest_modes)):
            x_pos = min(0.15 + (i / len(k_highest_modes)) * 0.9, 0.85)
            y_pos = min(
                abs(coefficients_mean_for_i_channel)[k_highest_modes[:, ch]][i] * 3.6, 0.81)
            positions.append((x_pos, y_pos))

        for (index, pos) in zip(k_highest_modes[..., ch], positions):
            ax_inset = ax[ch].inset_axes([pos[0], pos[1], 0.16, 0.16])
            vector_channel = vectors[index][ch] if n_basis > 1 else vectors[index][0]
            vector_channel = -vector_channel if coefficients_mean_for_i_channel[index] < 0 else vector_channel
            vector_data = np.flipud(np.transpose(vector_channel, (1, 0)))
            ax_inset.imshow(vector_data, origin='lower')
            ax_inset.set_title(r'$i$'+f'={index + 1}', fontsize=14)
            ax_inset.set_xticklabels([])
            ax_inset.set_yticklabels([])
            ax_inset.tick_params(labelsize=7)

            ax_inset.set_aspect('equal', adjustable='box')

            x_main = index
            y_main = np.abs(coefficients_mean[index, ch])
            coordsA = 'data'
            coordsB = 'axes fraction'
            for corner in [(0.5, 0)]:
                con = ConnectionPatch(
                    xyA=(x_main, y_main), coordsA=coordsA,
                    xyB=corner, coordsB=coordsB,
                    axesA=ax[ch], axesB=ax_inset,
                    color='black'
                )
                ax_inset.add_artist(con)

    plt.tight_layout()
    return fig


def plot_coefficients(branch_output_sample: np.ndarray, basis: np.ndarray, input_function_map: dict[str, float], target_labels: list[str]):
    # Branch output_sample: (n_channels, n_samples)
    # Basis: (n_samples, n_channels, coord1, coord2)
    parameters_map = [(k, v) for k, v in input_function_map.items()]

    colors = ['crimson', 'goldenrod', 'royalblue']

    fig_width, fig_height = 14, 5
    plt.subplots_adjust(wspace=0.1)
    wspace = 0.2
    subplot_width = (fig_width - (len(branch_output_sample) - 1)
                     * wspace) / len(branch_output_sample)

    fig, axes = plt.subplots(
        ncols=len(branch_output_sample), figsize=(fig_width, fig_height))

    for i, channel in enumerate(branch_output_sample):
        axes[i].bar(range(len(channel)), channel,
                    color=colors[i])
        axes[i].set_ylim([branch_output_sample.min() * 1.5,
                         branch_output_sample.max() * 1.3])
        axes[i].set_xlabel(r'$i$')
        axes[i].set_ylabel(r'$i$-th coefficient')
        axes[i].set_title(
            f'{target_labels[i]} ({parameters_map[0][0]}={parameters_map[0][1]:.1E}, {parameters_map[1][0]}={parameters_map[1][1]:.1E})')

        # positions = []
        # for j, coeff in enumerate(channel):
        #     x_pos = j / (subplot_width)
        #     y_pos = min(coeff / fig_height,
        #                 0.85) if coeff >= 1 else min(coeff,
        #                                              0.85) if coeff >= 0 else max(coeff / fig_height, 0)
        #     # min(coeff * 0.2, 0.8) if coeff >= 0 else max(-coeff * 2, 0)
        #     positions.append((x_pos, y_pos))

        # for (index, pos) in enumerate(positions):
        #     ax_inset = axes[i].inset_axes([pos[0], pos[1], 0.15, 0.15])
        #     vector_channel = basis[index][i]
        #     vector_data = np.flipud(np.transpose(vector_channel, (1, 0)))
        #     vector_data = vector_data if channel[index] >= 0 else - vector_data
        #     ax_inset.imshow(vector_data, origin='lower')
        #     ax_inset.set_title(r'$i$'+f'={index + 1}', fontsize=14)
        #     ax_inset.set_xticklabels([])
        #     ax_inset.set_yticklabels([])
        #     ax_inset.tick_params(labelsize=7)

        #     ax_inset.set_aspect('equal', adjustable='box')

        #     x_main = index
        #     y_main = channel[index]
        #     coordsA = 'data'
        #     coordsB = 'axes fraction'
        #     plot_pos = (0.5, 0) if channel[index] > 0 else (0.5, 1)
        #     for corner in [plot_pos]:
        #         con = ConnectionPatch(
        #             xyA=(x_main, y_main), coordsA=coordsA,
        #             xyB=corner, coordsB=coordsB,
        #             axesA=axes[i], axesB=ax_inset,
        #             color='black'
        #         )
        #         ax_inset.add_artist(con)

    fig.tight_layout()
    return fig
