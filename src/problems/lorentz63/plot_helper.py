import logging
from typing import Any
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.problems.lorentz63.plot_trajectories import plot_lorenz_trajectories, plot_lorenz_trajectories_3d
from src.problems.lorentz63.plot_basis import plot_basis_component, plot_basis_3d
from src.modules.models.deeponet.config import DataConfig, TestConfig
from src.problems.lorentz63.plot_coeffs import plot_coefficients, plot_coefficients_mean

logger = logging.getLogger(__file__)

def plot_trajectories_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    traj_1d_path = plot_path / '1d_trajectory'
    traj_3d_path = plot_path / '3d_trajectory'
    traj_1d_path.mkdir(exist_ok=True)
    traj_3d_path.mkdir(exist_ok=True)

    fig_1d_overlapped, pred_fig_3d_overlapped = plot_lorenz_trajectories(
        pred_trajectories=data['predictions'],
        true_trajectories=data['ground_truths'],
        trajectory_window=data['coordinates']['t'],
        initial_conditions=data['input_functions']['$\\mathbf{r_0}$'],
        num_to_plot=len(metadata['indices'])
    )
    fig_traj_1d_path = traj_1d_path / f"first_{len(metadata['indices'])}_trajectories.png"
    fig_traj_3d_path = traj_3d_path / f"first_{len(metadata['indices'])}_trajectories.png"
    if fig_1d_overlapped is not None:
        fig_1d_overlapped.savefig(fig_traj_1d_path)
    if pred_fig_3d_overlapped is not None:
        pred_fig_3d_overlapped.savefig(fig_traj_3d_path)
    plt.close()

    true_fig_3d_overlapped = plot_lorenz_trajectories_3d(
        trajectories=data['ground_truths'],
        trajectory_window=data['coordinates']['t'],
        initial_conditions=data['input_functions']['$\\mathbf{r_0}$'],
        num_to_plot=len(metadata['indices'])
    )
    fig_true_traj_3d_path = traj_3d_path / f"true_first_{len(metadata['indices'])}_trajectories.png"
    if true_fig_3d_overlapped is not None:
        true_fig_3d_overlapped.savefig(fig_true_traj_3d_path)
    plt.close()

    for i in range(len(metadata['indices'])):
        fig_1d, fig_3d = plot_lorenz_trajectories(
            pred_trajectories=data['predictions'][i][None, :, :],
            true_trajectories=data['ground_truths'][i][None, :, :],
            trajectory_window=data['coordinates']['t'],
            initial_conditions=data['input_functions']['$\\mathbf{r_0}$'][i],
        )
        fig_traj_1d_path = traj_1d_path / f'trajectory_{i+1}_trajectories.png'
        fig_traj_3d_path = traj_3d_path / f'trajectory_{i+1}_trajectories.png'

        if fig_1d is not None:
            fig_1d.savefig(fig_traj_1d_path)
        if fig_3d is not None:
            fig_3d.savefig(fig_traj_3d_path)
        plt.close()
    

def plot_basis_helper(data: dict[str, Any], data_cfg: DataConfig, plot_path: Path):
    basis_path_1d = plot_path / f"1d"
    basis_path_1d.mkdir(exist_ok=True)

    basis_path_3d = plot_path / f"3d"
    basis_path_3d.mkdir(exist_ok=True)

    mask_1 = len(data['bias']) > 1
    mask_2 = data['bias'].ndim > 1
    mask = mask_1 and mask_2
    if mask:
        fig_bias_1d = plot_basis_component(
            coords=data['coordinates']['t'],
            basis=data['bias'][0],
            index=0,
            target_labels=data_cfg.targets_labels,
        )
        fig_bias_1d_path = basis_path_1d / f"bias.png"
        fig_bias_1d.savefig(fig_bias_1d_path)
        plt.close()

        fig_bias_3d = plot_basis_3d(
            coords=data['coordinates']['t'],
            basis=data['bias'][0],
            index=0,
            target_labels=data_cfg.targets_labels,
        )
        fig_bias_3d_path = basis_path_3d / f"bias.png"
        fig_bias_3d.savefig(fig_bias_3d_path)
        plt.close()


    for i in tqdm(range(1, len(data['basis']) + 1), colour='blue'):
        fig_basis_1d = plot_basis_component(
            coords=data['coordinates']['t'],
            basis=data['basis'][i - 1],
            index=i,
            target_labels=data_cfg.targets_labels,
        )
        fig_basis_1d_path = basis_path_1d / f"vector_{i}.png"
        fig_basis_1d.savefig(fig_basis_1d_path)
        plt.close()

        if data['basis'][i - 1].shape[0] == 1:
            continue

        fig_basis_3d = plot_basis_3d(
            coords=data['coordinates']['t'],
            basis=data['basis'][i - 1],
            index=i,
            target_labels=data_cfg.targets_labels,
        )
        fig_basis_3d_path = basis_path_3d / f"vector_{i}.png"
        fig_basis_3d.savefig(fig_basis_3d_path)
        plt.close()

def plot_coefficients_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    for idx in metadata['indices']:
        param_val = metadata['values'][idx]
        fig_coeffs = plot_coefficients(
            branch_output_sample=data['coefficients'][idx],
            basis=data['basis'],
            input_function_map={
                data_cfg.input_functions[0]: param_val,
                },
            target_labels=data_cfg.targets_labels
        )
        # plt.show()
        # quit()
        file_name = f"r_0_indice_{idx}.png"
        fig_coeffs_path = plot_path / file_name
        fig_coeffs.savefig(fig_coeffs_path)
        plt.close()


def plot_coefficients_mean_helper(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig, plot_path: Path):
    if test_cfg.config is None:
        raise AttributeError(f"Missing attribute 'config'")
    fig_coeffs_mean = plot_coefficients_mean(
        vectors=data['basis'],
        coefficients=data['coefficients'],
        num_vectors_to_highlight=test_cfg.config['vectors_to_highlight'],
        target_labels=data_cfg.targets_labels
    )
    fig_coeffs_mean_path = plot_path / \
        f"coeffs_mean.png"
    fig_coeffs_mean.savefig(fig_coeffs_mean_path)
    plt.close()
