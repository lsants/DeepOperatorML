import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from src.problems.base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)

class Lorentz63Generator(BaseProblemGenerator):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def load_config(self) -> dict[str, Any]:
        return self.config
    
    def _get_input_functions(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        """Generate input functions to the operator (input to the branch).

        Args:
            data (np.ndarray): Original dataset is (num_trajectories * num_time_points, 7).
                               Here, each of the 7 columns is (t, x0, y0, z0, x, y, z)

        Returns:
            initial_conditions: Initial condition vector. Vector in space (x0, y0, z0) corresponding to the trajectorie's initial condition. 
        """
        unique_init_conds = []
        seen_init_conds = set()
        for row in data:
            init_cond_tuple = tuple(row[:3])
            if init_cond_tuple not in seen_init_conds:
                unique_init_conds.append(row[:3])
                seen_init_conds.add(init_cond_tuple)
        initial_conditions = np.array(unique_init_conds)
        return initial_conditions # (300, 3)
    
    def _get_coordinates(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        "Generate timesteps array from t=t0 to t=tf, in seconds."
        timestamps = np.unique(data[:, 3])
        sampled_indices = np.where(timestamps <= self.config['max_time'])
        traj_window = timestamps[sampled_indices]
        return traj_window # (501,)

    def _trajectories(self, data: np.ndarray, initial_conditions: np.ndarray, traj_window: np.ndarray) -> np.ndarray[Any, Any]:
        "Generate trajectories array."
        trajectories = data[ : , [-3, -2, -1]].reshape(len(initial_conditions), -1, 3)
        trajectories_sampled = trajectories[ : , : len(traj_window), :]
        return trajectories_sampled # (300, 501, 3)
    
    def generate(self):
        data = np.loadtxt(self.config['matlab_data_path'], delimiter=',')
        logger.info(f"Formatting...")
        initial_conditions = self._get_input_functions(data)
        trajectory_window = self._get_coordinates(data)
        trajectories = self._trajectories(data, initial_conditions, trajectory_window)

        logger.info(
            f"\nData shapes:\nInput functions (x0, y0, z0): {initial_conditions.shape},\nmin: ({initial_conditions[:, 0].min(axis=0):.2f}, {initial_conditions[:, 1].min(axis=0):.2f}, {initial_conditions[:, 2].min(axis=0):.2f})\nmax: ({initial_conditions[:, 0].max(axis=0):.2f}, {initial_conditions[:, 1].max(axis=0):.2f}, {initial_conditions[:, 2].max(axis=0):.2f})\nmean:({initial_conditions[:, 0].mean(axis=0):.2f}, {initial_conditions[:, 1].mean(axis=0):.2f}, {initial_conditions[:, 2].mean(axis=0):.2f})\nstd: ({initial_conditions[:, 0].std(axis=0):.2f}, {initial_conditions[:, 1].std(axis=0):.2f}, {initial_conditions[:, 2].std(axis=0):.2f})\nt=[{trajectory_window.min()}, {trajectory_window.max()}], {trajectory_window.shape}.\ntrajectories y: {trajectories.shape}\nx: min = {trajectories[:, :, 0].min():.2f}, max = {trajectories[:, :, 0].max():.2f}, mean = {trajectories[:, :, 0].mean():.2f}, std = {trajectories[:, :, 0].std():.2f}\ny: min = {trajectories[:, :, 1].min():.2f}, max = {trajectories[:, :, 1].max():.2f}, mean = {trajectories[:, :, 1].mean():.2f}, std = {trajectories[:, :, 1].std():.2f}\nz: min = {trajectories[:, :, 2].min():.2f}, max = {trajectories[:, :, 2].max():.2f}, mean = {trajectories[:, :, 2].mean():.2f}, std = {trajectories[:, :, 2].std():.2f}")
        
        metadata = {
            "initial_conditions": {
                "r0": {
                    "shape": [i for i in initial_conditions.shape],
                    "min":  f"({initial_conditions[:, 0].min(axis=0):.2f}, {initial_conditions[:, 1].min(axis=0):.2f}, {initial_conditions[:, 2].min(axis=0):.2f})",
                    "max":  f"({initial_conditions[:, 0].max(axis=0):.2f}, {initial_conditions[:, 1].max(axis=0):.2f}, {initial_conditions[:, 2].max(axis=0):.2f})",
                    "mean":  f"({initial_conditions[:, 0].mean(axis=0):.2f}, {initial_conditions[:, 1].mean(axis=0):.2f}, {initial_conditions[:, 2].mean(axis=0):.2f})",
                    "std":  f"({initial_conditions[:, 0].std(axis=0):.2f}, {initial_conditions[:, 1].std(axis=0):.2f}, {initial_conditions[:, 2].std(axis=0):.2f})",
                },
            },
            "time_coordinates": {
                "t": {
                    "shape": [i for i in trajectory_window.shape],
                    "min":  f"{trajectory_window.min():.2f}",
                    "max":  f"{trajectory_window.max():.2f}",
                    "mean": f"{trajectory_window.mean():.2f}",
                    "std":  f"{trajectory_window.std():.2f}"
                },
            },
            "trajectories": {
                "g_u": {
                    "shape": [i for i in trajectories.shape],
                    "min":  ', '.join([f'{i:.2f}' for i in trajectories.min(axis=(0, 1))]),
                    "max":  ', '.join([f'{i:.2f}' for i in trajectories.max(axis=(0, 1))]),
                    "mean": ', '.join([f'{i:.2f}' for i in trajectories.mean(axis=(0, 1))]),
                    "std":  ', '.join([f'{i:.2f}' for i in trajectories.std(axis=(0, 1))])
                }
            }
        }
        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path, 
            r_0=initial_conditions, 
            t=trajectory_window,
            g_u=trajectories
        )

        metadata_path = path.with_suffix('.yaml')  # Changes .npz to .yaml

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
