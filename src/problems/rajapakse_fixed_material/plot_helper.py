import logging
from typing import Any
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from src.modules.pipe.pipeline_config import DataConfig, TestConfig
from src.problems.rajapakse_fixed_material.plot_field import plot_2D_field
from src.problems.rajapakse_fixed_material.plot_axis import plot_axis
from src.problems.rajapakse_fixed_material.plot_basis import plot_basis
from src.problems.rajapakse_fixed_material.plot_coeffs import plot_coefficients, plot_coefficients_mean

logger = logging.getLogger(__file__)


def plot_planes_helper(data: dict[str, dict[str, Any]], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    mask = [i for i in metadata.keys() if i != 'percentiles']
    sample_map = metadata[mask[0]]
    percentiles = metadata['percentiles']
    for count, idx in tqdm(enumerate(sample_map['indices']), colour='green'):
        param_val = sample_map['values'][count]
        fig_plane = plot_2D_field(
            coords=data['coordinates'],
            truth_field=data['ground_truths'][idx],
            pred_field=data['predictions'][idx],
            input_function_value=param_val,
            input_function_labels=data_cfg.input_functions,
            target_labels=data_cfg.targets_labels
        )
        val_str = f"{param_val:.2f}"
        file_name = f"{percentiles[count]:.0f}_th_percentile_" + \
            'δ' + f"={val_str}.png"
        fig_plane_path = plot_path / file_name
        fig_plane.savefig(fig_plane_path)
        plt.close()


def plot_axis_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    mask = [i for i in metadata.keys() if i != 'percentiles']
    sample_map = metadata[mask[0]]
    for count, idx in tqdm(enumerate(sample_map['indices']), colour='green'):
        param_val = sample_map['values'][count]
        fig_axis = plot_axis(
            coords=data['coordinates'],
            truth_field=data['ground_truths'][idx],
            pred_field=data['predictions'][idx],
            param_map={rf'$\delta$': param_val},  # type: ignore
            target_labels=data_cfg.targets_labels
        )
        val_str = f"{param_val:.2f}"
        file_name = 'δ' + f"={val_str}.png"
        fig_axis_path = plot_path / file_name
        fig_axis.savefig(fig_axis_path)
        plt.close()


def plot_basis_helper(data: dict[str, Any], data_cfg: DataConfig, plot_path: Path):
    if data['bias'].ndim > 1:
        fig_bias = plot_basis(
            coords=data['coordinates'],
            basis=data['bias'],
            index=0,
            target_labels=data_cfg.targets_labels
        )
        fig_basis_path = plot_path / f"bias.png"
        fig_bias.savefig(fig_basis_path)
        plt.close()

    for i in tqdm(range(1, len(data['basis']) + 1), colour='blue'):
        fig_basis = plot_basis(
            coords=data['coordinates'],
            basis=data['basis'][i - 1],
            index=i,
            target_labels=data_cfg.targets_labels
        )
        fig_basis_path = plot_path / f"vector_{i}.png"
        fig_basis.savefig(fig_basis_path)
        plt.close()


def plot_coefficients_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    mask = [i for i in metadata.keys() if i != 'percentiles']
    sample_map = metadata[mask[0]]
    for count, idx in tqdm(enumerate(sample_map['indices']), colour='blue'):
        param_val = sample_map['values'][count]
        fig_coeffs = plot_coefficients(
            branch_output_sample=data['coefficients'][idx],
            basis=data['basis'],
            input_function_map={
                data_cfg.input_functions[0]: data['input_functions'][data_cfg.input_functions[0]][idx]},
            target_labels=data_cfg.targets_labels
        )
        val_str = f"{param_val:.2f}"
        file_name = r"coeffs_δ=" + f"{val_str}.png"
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
