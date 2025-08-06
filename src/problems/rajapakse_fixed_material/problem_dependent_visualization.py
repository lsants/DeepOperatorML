from __future__ import annotations
import yaml
import logging
import numpy as np
from typing import Any
from src.problems.rajapakse_fixed_material import postprocessing as ppr
from src.problems.rajapakse_fixed_material import plot_helper as helper
from src.modules.models.deeponet.config import DataConfig, TestConfig

logger = logging.getLogger(__file__)

def get_plotted_samples_indices(data_cfg: DataConfig, test_cfg: TestConfig) -> tuple[dict[str, dict[str, list[np.intp | float]]], np.ndarray]:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")
    percentile_sample_by_parameter = {}
    chosen_percentiles = min(test_cfg.config["percentiles"], len(
        data_cfg.split_indices[data_cfg.features[0] + '_test']))
    percentiles = np.linspace(
        0, 100, num=chosen_percentiles)
    for pos, parameter in enumerate(data_cfg.input_functions):
        indices = []
        targets = []
        for perc in percentiles:
            target = np.percentile(
                data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0] + '_test']][:, pos], perc)
            idx = np.argmin(
                np.abs(data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0] + '_test']][:, pos] - target))
            indices.append(idx)
            targets.append(target)
        percentile_sample_by_parameter[parameter] = {
            'indices': indices,
            'values': targets
        }
    return percentile_sample_by_parameter, percentiles

def get_plot_metatada(percentiles_sample_map: dict[str, dict[str, list[Any]]], percentiles: np.ndarray):
    metadata = {}
    parameters_info = percentiles_sample_map.copy()
    for param in parameters_info:
        parameters_info[param]['values'] = [
            float(f"{v:.3f}") for v in parameters_info[param]['values']]
        parameters_info[param]['indices'] = [
            int(i) for i in parameters_info[param]['indices']]
    metadata = {
        'percentiles': [round(perc) for perc in percentiles],
        **parameters_info
    }
    return metadata

def save_plot_metadata(metadata: dict[str, Any], save_path: str):
    with open(f'{save_path}/plot_metadata.yaml', mode='w') as file:
        yaml.safe_dump(metadata, file, allow_unicode=True)


def run_problem_specific_plotting(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig):
    if test_cfg.problem is None:
        raise AttributeError(f"'Problem' attribute is missing.")
    plots_path = test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'plots'
    percentile_sample_by_parameter, percentiles = get_plotted_samples_indices(
        data_cfg=data_cfg, test_cfg=test_cfg)
    metadata = get_plot_metatada(percentile_sample_by_parameter, percentiles)

    save_plot_metadata(metadata, str(plots_path))

    plane_plots_path = plots_path / 'plane_plots'
    axis_plots_path = plots_path / 'axis_plots'
    basis_plots_path = plots_path / 'basis_plots'
    coefficients_plots_path = plots_path / 'coefficients_plots'

    for path in [plane_plots_path, axis_plots_path, basis_plots_path, coefficients_plots_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    if test_cfg.config['plot_plane']:
        helper.plot_planes_helper(
            data=data, 
            data_cfg=data_cfg, 
            metadata=metadata, 
            plot_path=plane_plots_path
        )

    if test_cfg.config['plot_axis']:
        helper.plot_axis_helper(
            data=data,
            data_cfg=data_cfg,
            metadata=metadata,
            plot_path=axis_plots_path
        )

    if test_cfg.config['plot_basis']:
        helper.plot_basis_helper(
            data=data,
            data_cfg=data_cfg,
            plot_path=basis_plots_path
        )

    if test_cfg.config['plot_coefficients']:
        helper.plot_coefficients_helper(
            data=data,
            data_cfg=data_cfg,
            metadata=metadata,
            plot_path=coefficients_plots_path
        )

    if test_cfg.config['plot_coefficients_mean']:
        helper.plot_coefficients_mean_helper(
            data=data,
            data_cfg=data_cfg,
            test_cfg=test_cfg,
            plot_path=coefficients_plots_path
        )


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig):
    input_functions = ppr.get_input_functions(data_cfg)
    coordinates = ppr.get_coordinates(data_cfg)
    output_data = ppr.get_output_data(test_cfg)
    ground_truths = ppr.format_target(output_data[data_cfg.targets[0]], data_cfg)
    predictions = ppr.format_target(output_data['predictions'], data_cfg)
    coefficients = ppr.reshape_coefficients(
        output_data['branch_output'], data_cfg, test_cfg)
    basis = ppr.reshape_basis(output_data['trunk_output'], data_cfg, test_cfg)
    bias = ppr.format_bias(output_data['bias'], data_cfg, test_cfg)

    data = {
        'input_functions': input_functions,
        'coordinates': coordinates,
        'output_data': output_data,
        'ground_truths': ground_truths,
        'predictions': predictions,
        'coefficients': coefficients,
        'basis': basis,
        'bias': bias,
    }

    run_problem_specific_plotting(
        data=data, data_cfg=data_cfg, test_cfg=test_cfg)
