from __future__ import annotations
import time
import yaml
import logging
import numpy as np
from ..base_generator import BaseProblemGenerator
from typing import Any
from pathlib import Path
from ...modules.utilities.sampling_functions import mesh_rescaling, numpy_random_open_0_1

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
        """Generate the branch data (operator parameters) by sampling N values for the load magnitude F,
        and then attaching the fixed material parameters mu and nu.

        Returns:
            array: Shape (N, 3) with columns [F, mu, nu], where N is the number of samples.
        """

        log_mu_samples = self.config["MU_MIN"] + numpy_random_open_0_1(
            self.config["N"]) * (self.config["MU_MAX"] - self.config["MU_MIN"])
        F_samples = -np.array(10**self.config["F"])
        mu_samples = 10**log_mu_samples
        nu_samples = numpy_random_open_0_1(
            self.config["N"]) * (self.config["NU_MAX"] - self.config["NU_MIN"])
        return F_samples, mu_samples, nu_samples

    def _get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the trunk data by defining a 3d grid in cartesian coordinates.
        Returns:
            tuple: [x, y, z]
        """
        x_field = np.linspace(
            self.config["X_MIN"]+1e-6, self.config["X_MAX"] - 1e-6, self.config["N_X"])
        y_field = np.linspace(
            self.config["Y_MIN"]+1e-6, self.config["Y_MAX"] - 1e-6, self.config["N_Y"])
        z_field = np.linspace(
            self.config["Z_MIN"]+1e-6, self.config["Z_MAX"] - 1e-6, self.config["N_Z"])
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
            input_functions (tuple): Tuple (F, mu, nu) each of shape (N,), where N is the total
                                    number of branch samples (the Cartesian product of sensor values).
            x_field (array): 1D array of x coordinates.
            y_field (array): 1D array of y coordinates.
            z_field (array): 1D array of z coordinates.

        Returns:
            u (ndarray): Array of shape (N, n_x, n_y, n_z, 3) containing the displacement field.
            duration (float): Computation time in seconds.
        """
        start = time.perf_counter_ns()

        if self.config["LOAD_DIRECTION"] == 'x':
            d = 0
        elif self.config["LOAD_DIRECTION"] == 'y':
            d = 1
        elif self.config["LOAD_DIRECTION"] == 'z':
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
            x_field_transformed, self.config["SCALER"])
        y_field = mesh_rescaling(
            y_field_transformed, self.config["SCALER"])
        z_field = mesh_rescaling(
            z_field_transformed, self.config["SCALER"])

        logger.info(f"Generating...")
        displacements, duration = self._influencefunc(
            F, mu, nu, x_field, y_field, z_field)

        scaler_parameter = self.config["SCALER"]

        logger.info(
            f"Runtime for computing Kelvin solution: {duration:.3f} ms\nData shapes:\nInput functions (F, mu, nu): {F.shape}, {mu.shape}, {nu.shape}\nDisplacements u: {displacements.shape}\nx: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}\nLoad magnitude = {F:.3E}\nShear modulus min = {mu.min():.3E}, max = {mu.max():.3E}\nPoisson's ratio min = {nu.min():.3f}, max = {nu.max():.3f}\nx: min = {x_field.min():3f}, max = {x_field.max():.3f}\nx: mean = {x_field.mean():3f}, std = {x_field.std():.3f}\ny: min = {y_field.min():3f}, max = {y_field.max():.3f}\ny: mean = {y_field.mean():3f}, std = {y_field.std():.3f}\nz: min = {z_field.min():3f}, max = {z_field.max():.3f}\nz: mean = {z_field.mean():3f}, std = {z_field.std():.3f}\ng_u: min = {displacements.min():3f}, max = {displacements.max():.3f}\ng_u: mean = {displacements.mean():3f}, std = {displacements.std():.3f}\n scaling parameter = {scaler_parameter:.3f}")

        # --- Metadata Collection ---
        metadata = {
            "runtime_ms": f"{duration:.3f}",
            "parameters": {
                # .item() for scalar array
                "load_magnitude": f"{F.item():.3E}" if F.size == 1 else "N/A",
                "shear_modulus": {
                    "min": f"{mu.min():.3E}",
                    "max": f"{mu.max():.3E}",
                    "mean": f"{mu.mean():.3E}",
                    "std": f"{mu.std():.3E}"
                },
                "poissons_ratio": {
                    "min": f"{nu.min():.3f}",
                    "max": f"{nu.max():.3f}",
                    "mean": f"{nu.mean():.3f}",
                    "std": f"{nu.std():.3f}"
                },
                "scaling_parameter": f"{scaler_parameter:.3f}"
            },
            "coordinate_statistics": {
                "x_field": {
                    "min": f"{x_field.min():.3f}",
                    "max": f"{x_field.max():.3f}",
                    "mean": f"{x_field.mean():.3f}",
                    "std": f"{x_field.std():.3f}"
                },
                "y_field": {
                    "min": f"{y_field.min():.3f}",
                    "max": f"{y_field.max():.3f}",
                    "mean": f"{y_field.mean():.3f}",
                    "std": f"{y_field.std():.3f}"
                },
                "z_field": {
                    "min": f"{z_field.min():.3f}",
                    "max": f"{z_field.max():.3f}",
                    "mean": f"{z_field.mean():.3f}",
                    "std": f"{z_field.std():.3f}"
                }
            },
            "displacement_statistics": {
                "g_u": {
                    "min": f"{displacements.min():.4E}",
                    "max": f"{displacements.max():.4E}",
                    "mean": f"{displacements.mean():.4E}",
                    "std": f"{displacements.std():.4E}"
                }
            }
        }
        path = Path(self.config["DATA_PATH"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(path, mu=mu, nu=nu, x=x_field, y=y_field,
                 z=z_field, g_u=displacements, c=scaler_parameter)

        metadata_path = path.with_suffix('.yaml')  # Changes .npz to .yaml

        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
