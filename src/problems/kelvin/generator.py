from __future__ import annotations
import time
import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from src.problems.base_generator import BaseProblemGenerator
from src.modules.utilities.sampling_functions import mesh_rescaling, numpy_random_open_0_1

logger = logging.getLogger(__name__)

class KelvinProblemGenerator(BaseProblemGenerator):
    def __init__(self, config: str | dict[str, Any]):
        super().__init__(config=config)
        if isinstance(config, (str, Path)):
            self.config_path = config
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = config

    def load_config(self):
        if self.config_path:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self.config

    def _get_input_functions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the branch data (operator parameters) by sampling N values for the uniform material parameters mu and nu.

        Returns:
            array: Shape (N, 3) with columns [F, mu, nu], where N is the number of samples.
        """

        log_mu_samples = self.config["mu_min"] + numpy_random_open_0_1(
            self.config["N"]) * (self.config["mu_max"] - self.config["mu_min"])
        F_samples = -np.array(10**self.config["F"])
        mu_samples = 10**log_mu_samples
        nu_samples = numpy_random_open_0_1(
            self.config["N"]) * (self.config["nu_max"] - self.config["nu_min"])
        return F_samples, mu_samples, nu_samples

    def _get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the trunk data by defining a 3d grid in cartesian coordinates.
        Returns:
            tuple: [x, y, z]
        """
        x_field = np.linspace(
            self.config["x_min"]+1e-6, self.config["x_max"] - 1e-6, self.config["N_x"])
        y_field = np.linspace(
            self.config["y_min"]+1e-6, self.config["y_max"] - 1e-6, self.config["N_y"])
        z_field = np.linspace(
            self.config["z_min"]+1e-6, self.config["z_max"] - 1e-6, self.config["N_z"])
        return x_field, y_field, z_field

    def _influencefunc(self, F, mu, nu, x_field, y_field, z_field) -> tuple[np.ndarray, float]:
        """
        Compute the Kelvin solution in Cartesian coordinates in a fully vectorized way.
        This version assumes that the branch inputs (F, mu, nu) are sampled N time each.
        For each branch sample (F, mu, nu) and for each grid point (x, y, z), the displacement
        vector is computed using Kelvin's formula:

        u_i = (F / (16 * π * mu * (1 - nu))) * [ (3 - 4*nu)*δ_{i,d} / r  +  (x_i*x_d) / r³ ]

        where r = sqrt(x² + y² + z²), and d is the index corresponding to the load direction.

        Args:
            F (array): Single element force 1D array.
            mu (array): 1D array of shear modulus samples.
            nu (array): 1D array of Poisson's ratio samples.
            x_field (array): 1D array of x coordinates.
            y_field (array): 1D array of y coordinates.
            z_field (array): 1D array of z coordinates.

        Returns:
            u (ndarray): Array of shape (N, n_x, n_y, n_z, 3) containing the displacement field.
            duration (float): Computation time in seconds.
        """
        start = time.perf_counter_ns()

        if self.config["load_direction"] == 'x':
            d = 0
        elif self.config["load_direction"] == 'y':
            d = 1
        elif self.config["load_direction"] == 'z':
            d = 2
        else:
            raise ValueError(
                "Invalid load direction. Must be 'x', 'y', or 'z'.")

        X, Y, Z = np.meshgrid(x_field, y_field, z_field, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)

        r_vals = np.linalg.norm(coords, axis=-1)  # Shape: (n_x, n_y, n_z)
        r_b = r_vals[None, ...]  # Shape: (1, n_x, n_y, n_z)

        const = F / (16 * np.pi * mu * (1 - nu))
        const = const[:, None, None, None]

        factor = (3 - 4 * nu)
        factor = factor[:, None, None, None]

        coords_b = coords[None, ...]

        delta = np.zeros(3)
        delta[d] = 1
        delta = delta.reshape(1, 1, 1, 1, 3)

        r_inv = 1 / r_b
        r_inv3 = 1 / (r_b ** 3)

        term1 = (factor[..., None] * delta) * r_inv[..., None]

        coord_d = coords_b[..., d: d+1]  # shape: (1, n_x, n_y, n_z, 1)
        term2 = (coords_b * coord_d) * r_inv3[..., None]

        u = const[..., None] * (term1 + term2)

        end = time.perf_counter_ns()
        duration = (end - start) / 1e6
        return u, duration

    def generate(self):
        F, mu, nu = self._get_input_functions()
        coordinates = self._get_coordinates()
        x_field_transformed, y_field_transformed, z_field_transformed = coordinates

        x_field = mesh_rescaling(
            x_field_transformed, self.config["scaler"])
        y_field = mesh_rescaling(
            y_field_transformed, self.config["scaler"])
        z_field = mesh_rescaling(
            z_field_transformed, self.config["scaler"])

        logger.info(f"Generating...")
        displacements, duration = self._influencefunc(
            F, mu, nu, x_field, y_field, z_field)

        scaler_parameter = self.config["scaler"]

        logger.info(
            f"Runtime for computing Kelvin solution: {duration:.3f} ms\nData shapes:\nInput functions (F, mu, nu): {F.shape}, {mu.shape}, {nu.shape}\nDisplacements u: {displacements.shape}\nx: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}\nLoad magnitude = {F:.3E}\nShear modulus min = {mu.min():.3E}, max = {mu.max():.3E}\nPoisson's ratio min = {nu.min():.3f}, max = {nu.max():.3f}\nx: min = {x_field.min():3f}, max = {x_field.max():.3f}\nx: mean = {x_field.mean():3f}, std = {x_field.std():.3f}\ny: min = {y_field.min():3f}, max = {y_field.max():.3f}\ny: mean = {y_field.mean():3f}, std = {y_field.std():.3f}\nz: min = {z_field.min():3f}, max = {z_field.max():.3f}\nz: mean = {z_field.mean():3f}, std = {z_field.std():.3f}\ng_u: min = {displacements.min():3f}, max = {displacements.max():.3f}\ng_u: mean = {displacements.mean():3f}, std = {displacements.std():.3f}\n scaling parameter = {scaler_parameter:.3f}")

        # --- Metadata Collection ---
        metadata = {
            "runtime_ms": f"{duration:.3f}",
            "parameters": {
                # .item() for scalar array
                "load_magnitude": f"{F.item():.3E}" if F.size == 1 else "N/A",
                "shear_modulus": {
                    "shape": [i for i in mu.shape],
                    "min":  f"{mu.min():.3E}",
                    "max":  f"{mu.max():.3E}",
                    "mean": f"{mu.mean():.3E}",
                    "std":  f"{mu.std():.3E}"
                },
                "poissons_ratio": {
                    "shape": [i for i in mu.shape],
                    "min":  f"{nu.min():.3f}",
                    "max":  f"{nu.max():.3f}",
                    "mean": f"{nu.mean():.3f}",
                    "std":  f"{nu.std():.3f}"
                },
                "scaling_parameter": f"{scaler_parameter:.3f}"
            },
            "coordinate_statistics": {
                "x": {
                    "shape": [i for i in x_field.shape],
                    "min":  f"{x_field.min():.3f}",
                    "max":  f"{x_field.max():.3f}",
                    "mean": f"{x_field.mean():.3f}",
                    "std":  f"{x_field.std():.3f}"
                },
                "y": {
                    "shape": [i for i in y_field.shape],
                    "min":  f"{y_field.min():.3f}",
                    "max":  f"{y_field.max():.3f}",
                    "mean": f"{y_field.mean():.3f}",
                    "std":  f"{y_field.std():.3f}"
                },
                "z": {
                    "shape": [i for i in z_field.shape],
                    "min":  f"{z_field.min():.3f}",
                    "max":  f"{z_field.max():.3f}",
                    "mean": f"{z_field.mean():.3f}",
                    "std":  f"{z_field.std():.3f}"
                }
            },
            "displacement_statistics": {
                "g_u": {
                    "shape": [i for i in displacements.shape],
                    "min":  ', '.join([f'{i:.4E}' for i in  displacements.min(axis=(0, 1, 2, 3))]),
                    "max":  ', '.join([f'{i:.4E}' for i in  displacements.max(axis=(0, 1, 2, 3))]),
                    "mean": ', '.join([f'{i:.4E}' for i in  displacements.mean(axis=(0, 1, 2, 3))]),
                    "std":  ', '.join([f'{i:.4E}' for i in  displacements.std(axis=(0, 1, 2, 3))])
                }
            }
        }
        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(path, mu=mu, nu=nu, x=x_field, y=y_field,
                 z=z_field, g_u=displacements, c=scaler_parameter)

        metadata_path = path.with_suffix('.yaml')  # Changes .npz to .yaml

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
