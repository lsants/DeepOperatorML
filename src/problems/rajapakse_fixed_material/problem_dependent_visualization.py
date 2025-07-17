from __future__ import annotations
import yaml
import logging
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from src.modules.pipe.pipeline_config import DataConfig, TestConfig
from src.modules.plotting import plot_2D_field, plot_axis, plot_basis, plot_coefficients_with_basis, plot_coefficients
logger = logging.getLogger(__file__)


def get_output_data(test_cfg: TestConfig) -> dict[str, np.ndarray]:
    if test_cfg.problem is None:
        raise ValueError(f"Problem name must be set in TestConfig.")
    base_dir = Path(__file__).parent.parent.parent.parent
    output_data_path = base_dir / test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'aux' / 'output_data.npz'
    output_data = {i: j for i, j in np.load(output_data_path).items()}
    return output_data


def get_input_functions(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    delta = raw_data['delta']
    input_functions = {data_cfg.input_functions[0]: delta}
    return input_functions


def get_coordinates(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    r = raw_data['r']
    z = raw_data['z']
    coordinates = {'r': r, 'z': z}
    return coordinates


def format_target(displacements: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)
    displacements = displacements.reshape(
        -1,
        data_cfg.shapes[data_cfg.targets[0]][-1],
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][1],
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][2],
    )
    displacements_flipped = np.flip(displacements, axis=2)
    displacements_full = np.concatenate(
        (displacements, displacements_flipped), axis=2)
    return displacements_full


def reshape_coefficients(branch_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    return branch_out.reshape(
        -1,
        test_cfg.model.rescaling.num_basis_functions,  # type: ignore
        data_cfg.shapes[data_cfg.targets[0]][-1]
    )


def reshape_basis(trunk_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)

    return trunk_out.reshape(
        test_cfg.model.rescaling.num_basis_functions,  # type: ignore
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][1],
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][2],
        data_cfg.shapes[data_cfg.targets[0]][-1]
    )


def get_plotted_samples_indices(data_cfg: DataConfig, test_cfg: TestConfig) -> dict[int, list[np.intp]]:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")
    selected_indices = {}
    chosen_percentiles = min(test_cfg.config["percentiles"], len(
        data_cfg.split_indices[data_cfg.features[0].upper() + '_test']))
    percentiles = np.linspace(
        0, 100, num=chosen_percentiles)
    for parameter in range(data_cfg.shapes[data_cfg.features[0]][1]):
        indices = []
        for perc in percentiles:
            target = np.percentile(
                data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0].upper() + '_test']][:, parameter], perc)
            idx = np.argmin(
                np.abs(data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0].upper() + '_test']][:, parameter] - target))
            indices.append(idx)
        selected_indices[parameter] = indices
        logger.info(
            f"\nSelected indices for input parameter {parameter}: {indices}\n")
        logger.info(
            f"\nSelected values for input parameter {parameter}: {data_cfg.data[data_cfg.features[0]][indices]}\n")
    return selected_indices


def run_problem_specific_plotting(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig):
    plots_path = test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'plots'  # type: ignore
    samples_by_percentiles = get_plotted_samples_indices(
        data_cfg=data_cfg, test_cfg=test_cfg)
    plane_plots_path = plots_path / 'plane_plots'
    axis_plots_path = plots_path / 'axis_plots'
    basis_plots_path = plots_path / 'basis_plots'
    coefficients_plots_path = plots_path / 'coefficients_plots'

    for path in [plane_plots_path, axis_plots_path, basis_plots_path, coefficients_plots_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    for parameter, indices in tqdm(samples_by_percentiles.items(), colour='blue'):
        for count, idx in tqdm(enumerate(indices), colour='green'):
            param_val = data['input_functions'][data_cfg.input_functions[0]][idx]
            if test_cfg.config['plot_plane']:
                fig_plane = plot_2D_field(
                    coords=data['coordinates'],
                    truth_field=data['ground_truths'][idx],
                    pred_field=data['predictions'][idx],
                    input_function_value=param_val,
                    input_function_labels=data_cfg.input_functions,
                    target_labels=data_cfg.targets_labels
                )
                val_str = f"{param_val:.2f}"
                fig_plane_path = plane_plots_path / \
                    f"{count * 10}_th_percentile_{data_cfg.input_functions[parameter]}={val_str}.png"
                fig_plane.savefig(fig_plane_path)
                plt.close()
    for parameter, indices in tqdm(selected_indices.items(), colour='blue'):
        for count, idx in tqdm(enumerate(indices), colour='green'):
            if test_cfg.config['plot_axis']:
                fig_axis = plot_axis(
                    coords=data['coordinates'],
                    truth_field=data['ground_truths'][idx],
                    pred_field=data['predictions'][idx],
                    param_map={data_cfg.input_functions[parameter]: param_val},
                    target_labels=data_cfg.targets_labels
                )
                fig_axis_path = axis_plots_path / \
                    f"axis_parameter_{parameter}_for_param_{param_val}.png"
                fig_axis.savefig(fig_axis_path)

    if test_cfg.config['plot_basis']:
        for i in tqdm(range(1, len(data['basis']) + 1), colour='blue'):
            fig_basis = plot_basis(
                data['coordinates'],
                data['basis'][i - 1],
                index=i,
                basis_config=test_cfg.model.output.handler_type,  # type: ignore
                strategy=test_cfg.model.strategy.name,  # type: ignore
                param_val=None,
                output_keys=data_cfg.targets_labels
            )
            fig_basis_path = basis_plots_path / f"mode_{i}.png"
            fig_basis.savefig(fig_basis_path)
            plt.close()


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig):
    input_functions = get_input_functions(data_cfg)
    coordinates = get_coordinates(data_cfg)
    output_data = get_output_data(test_cfg)
    ground_truths = format_target(output_data[data_cfg.targets[0]], data_cfg)
    predictions = format_target(output_data['predictions'], data_cfg)
    coefficients = reshape_coefficients(
        output_data['branch_output'], data_cfg, test_cfg)
    basis = reshape_basis(output_data['trunk_output'], data_cfg, test_cfg)

    data = {
        'input_functions': input_functions,
        'coordinates': coordinates,
        'output_data': output_data,
        'ground_truths': ground_truths,
        'predictions': predictions,
        'coefficients': coefficients,
        'basis': basis
    }

    run_problem_specific_plotting(
        data=data, data_cfg=data_cfg, test_cfg=test_cfg)
