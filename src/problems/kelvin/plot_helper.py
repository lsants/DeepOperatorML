import logging
from typing import Any
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.problems.kelvin.plot_field import plot_field
from src.problems.kelvin.plot_basis import plot_basis
from src.modules.pipe.pipeline_config import DataConfig, TestConfig
from src.problems.kelvin.plot_coeffs import plot_coefficients, plot_coefficients_mean

logger = logging.getLogger(__file__)


def plot_planes_helper(data: dict[str, dict[str, Any]], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    percentiles = metadata['percentiles']
    for_filename = { "$\\nu$": 'ν' , 
                    "$\\mu$": 'μ'}

    planes = ['xy', 'xz', 'yz']
    variables = ['predictions', 'truths', 'rel_errors']

    for p, param in enumerate(data_cfg.input_functions):
        sample_map = metadata[param]
        for plane in planes:
            plane_path = plot_path / f"{plane}_plane"
            plane_path.mkdir(exist_ok=True)

            for var in variables:
                var_path = plane_path / var
                var_path.mkdir(exist_ok=True)
                
                for c, idx in tqdm(enumerate(sample_map['indices']), colour='green'):
                    branch_sample_searched = data_cfg.data[data_cfg.features[0]][data_cfg.split_indices['xb_test']][idx]
                    fig_plane = plot_field(
                        coords=data['coordinates'],
                        truth_field=data['ground_truths'][idx],
                        pred_field=data['predictions'][idx],
                        input_function_values=branch_sample_searched,
                        input_function_labels=data_cfg.input_functions,
                        target_labels=data_cfg.targets_labels,
                        plot_plane=plane,
                        plotted_variable=var
                    )
                    val_str = f"{branch_sample_searched[p]:.1E}"
                    file_name = f"{for_filename[param]}_{percentiles[c] / 100:.0%}_perc_{val_str}.png"
                    fig_plane_path = var_path / file_name
                    fig_plane.savefig(fig_plane_path)
                    plt.close()


def plot_basis_helper(data: dict[str, Any], data_cfg: DataConfig, plot_path: Path):
    planes = ['xy', 'xz', 'yz']
    for plane in planes:
        plane_path = plot_path / f"{plane}_plane"
        plane_path.mkdir(exist_ok=True)
        if data['bias'].ndim > 1:
            fig_bias = plot_basis(
                coords=data['coordinates'],
                basis=data['bias'],
                index=0,
                target_labels=data_cfg.targets_labels,
                plot_plane=plane
            )
            fig_basis_path = plot_path / plane_path / f"bias.png"
            fig_bias.savefig(fig_basis_path)
            plt.close()

        for i in tqdm(range(1, len(data['basis']) + 1), colour='blue'):
            fig_basis = plot_basis(
                coords=data['coordinates'],
                basis=data['basis'][i - 1],
                index=i,
                target_labels=data_cfg.targets_labels,
                plot_plane=plane
            )
            fig_basis_path = plane_path / f"vector_{i}.png"
            fig_basis.savefig(fig_basis_path)
            plt.close()

def plot_coefficients_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    mask = [i for i in metadata.keys() if i != 'percentiles']
    for_filename = { "$\\nu$": 'ν' , 
                    "$\\mu$": 'μ'
    }
    percentiles = metadata['percentiles']
    for param in mask:
        sample_map = metadata[param]
        for count, idx in enumerate(sample_map['indices']):
            param_val_0 = metadata[mask[0]]['values'][count]
            param_val_1 = metadata[mask[1]]['values'][count]
            fig_coeffs = plot_coefficients(
                branch_output_sample=data['coefficients'][idx],
                basis=data['basis'],
                input_function_map={
                    data_cfg.input_functions[0]: param_val_0,
                    data_cfg.input_functions[1]: param_val_1,
                    },
                target_labels=data_cfg.targets_labels
            )
            val_str = f"{sample_map['values'][count]:.3E}"
            file_name = f"{for_filename[param]}_{percentiles[count]}_perc_{val_str}.png"
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
