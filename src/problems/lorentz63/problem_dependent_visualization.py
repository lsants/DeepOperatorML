from __future__ import annotations
import yaml
import logging
import numpy as np
from typing import Any
from src.problems.lorentz63 import postprocessing as ppr
from src.problems.lorentz63 import plot_helper as helper
from src.modules.pipe.pipeline_config import DataConfig, TestConfig

logger = logging.getLogger(__file__)

def get_plotted_samples_indices(data_cfg: DataConfig, test_cfg: TestConfig) -> dict[str, list[str | np.ndarray]]:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")
    indices = []
    trajs = []
    num_trajectories = test_cfg.config["trajectories_to_plot"]
    for pos, traj in zip(range(num_trajectories), data_cfg.data['xb']):
        indices.append(pos)
        trajs.append(traj)
    sampled_trajectories = {
        'indices': indices,
        'values': trajs
    }
    return sampled_trajectories

def get_plot_metatada(sampled_trajectories: dict[str, dict[str, list[Any]]]):
    metadata = {}

    parameters_info = sampled_trajectories.copy()
    parameters_info['values'] = [
        v.tolist() for v in parameters_info['values']]
    metadata = {
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
    sampled_trajectories = get_plotted_samples_indices(
        data_cfg=data_cfg, test_cfg=test_cfg)
    metadata = get_plot_metatada(sampled_trajectories=sampled_trajectories)

    save_plot_metadata(metadata, str(plots_path))

    trajectories_plots_path = plots_path / 'trajectories_plots'
    basis_plots_path = plots_path / 'basis_plots'
    coefficients_plots_path = plots_path / 'coefficients_plots'

    for path in [coefficients_plots_path, trajectories_plots_path, basis_plots_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    if test_cfg.config['plot_trajectories']:
        helper.plot_trajectories_helper(
            data=data, 
            data_cfg=data_cfg, 
            metadata=metadata, 
            plot_path=trajectories_plots_path,
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
    basis = ppr.reshape_basis(output_data['trunk_output'], data_cfg, test_cfg)
    coefficients = ppr.reshape_coefficients(
        output_data['branch_output'], data_cfg, test_cfg)
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
