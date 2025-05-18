from __future__ import annotations
import time
import yaml
import logging
import numpy as np
from ..base_generator import BaseProblemGenerator
from typing import Any
from pathlib import Path

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

        log_F_samples = self.config["F_MIN"] + np.random.rand(self.config["N"]) * (self.config["F_MAX"] - self.config["F_MIN"])
        log_mu_samples = self.config["MU_MIN"] + np.random.rand(self.config["N"]) * (self.config["MU_MAX"] - self.config["MU_MIN"])
        F_samples = 10**log_F_samples
        mu_samples = 10**log_mu_samples
        nu_samples = self.config["NU_MIN"] + np.random.rand(self.config["N"]) * (self.config["NU_MAX"] - self.config["NU_MIN"])
        return F_samples, mu_samples, nu_samples
    
    def _get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the trunk data by defining a 3d grid in cartesian coordinates.
        Returns:
            tuple: [x, y, z]
        """
        x_field = np.linspace(self.config["X_MIN"], self.config["X_MAX"], self.config["N_X"])
        y_field = np.linspace(self.config["Y_MIN"], self.config["Y_MAX"], self.config["N_Y"])
        z_field = np.linspace(self.config["Z_MIN"], self.config["Z_MAX"], self.config["N_Z"])
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
            raise ValueError("Invalid load direction. Must be 'x', 'y', or 'z'.")

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
        delta = delta.reshape(1, 1, 1, 1 ,3)

        r_inv = 1 / r_b
        r_inv3 = 1 / (r_b ** 3)
        
        term1 = (factor[..., None] * delta) * r_inv[..., None]
        
        coord_d = coords_b[..., d : d+1]  # shape: (1, n_x, n_y, n_z, 1)
        term2 = (coords_b * coord_d) * r_inv3[..., None]
        
        u = const[..., None] * (term1 + term2)
        
        end = time.perf_counter_ns()
        duration = (end - start) / 1e6
        return u, duration
    
    def generate(self):
        F, mu, nu = self._get_input_functions()
        coordinates = self._get_coordinates()
        x_field, y_field, z_field = coordinates

        logger.info(f"Generating...")
        displacements, duration = self._influencefunc(F, mu, nu, x_field, y_field, z_field)

        #2D if one coord has one point
        # displacements = np.squeeze(displacements, axis=2)

        logger.info(f"Runtime for computing Kelvin solution: {duration:.3f} ms")
        logger.info(f"\nData shapes:")
        logger.info(f"   Input functions (F, mu, nu): {F.shape}, {mu.shape}, {nu.shape}")
        logger.info(f"   Displacements u: {displacements.shape}")
        logger.info(f"   x: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}")
        logger.info(f"\nLoad magnitude min = {F.min():.3f}, max = {F.max():.3f}")
        logger.info(f"\nShear modulus magnitude min = {mu.min():.3f}, max = {mu.max():.3f}")
        logger.info(f"\nPoisson's ratio min = {nu.min():.3f}, max = {nu.max():.3f}")
        logger.info(f"x: min = {x_field.min():3f}, max = {x_field.max():.3f}")
        logger.info(f"y: min = {y_field.min():3f}, max = {y_field.max():.3f}")
        logger.info(f"z: min = {z_field.min():3f}, max = {z_field.max():.3f}")

        path = Path(self.config["DATA_PATH"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, F=F, mu=mu, nu=nu, x=x_field, y=y_field, z=z_field, g_u=displacements)
        logger.info(f"Saved data at {path}")