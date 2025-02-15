import time
import logging
import numpy as np
from tqdm.auto import tqdm
from .data_generation_base import Datagen

logger = logging.getLogger(__name__)

class KelvinsProblemDeterministic(Datagen):
    def __init__(self, data_size, material_params, load_params, mesh_params, problem_setup):
        super().__init__(data_size, material_params, load_params, mesh_params, problem_setup)

    def _get_input_functions(self):
        """Generate the branch data (operator parameters) by sampling N values for the load magnitude F,
        and then attaching the fixed material parameters mu and nu.

        Returns:
            array: Shape (N, 3) with columns [F, mu, nu].
        """
        N = self.data_size[0]
        mu, nu = self.material_params
        F_min, F_max = self.load_params
        F_samples = F_min + np.random.rand(N) * (F_max - F_min)
        mu = np.full(N, mu)
        nu = np.full(N, nu)
        return F_samples, mu, nu
    
    def _get_coordinates(self):
        """Generate the trunk data by defining a 3d grid in cartesian coordinates.
        mesh_params = (x_min, x_max, y_min, y_max, z_min, z_max)

        Returns:
            tuple: [x, y, z]
        """
        _, n_x, n_y, n_z = self.data_size
        x_min, x_max, y_min, y_max, z_min, z_max = self.mesh_params
        x_field = np.linspace(x_min, x_max, n_x)
        y_field = np.linspace(y_min, y_max, n_y)
        z_field = np.linspace(z_min, z_max, n_z)
        return x_field, y_field, z_field
        
    def _influencefunc(self, input_functions, x_field, y_field, z_field):
        """
        Compute the Kelvin solution in Cartesian coordinates in a fully vectorized way.
        This version assumes that the branch inputs (F, mu, nu) have been generated as the
        Cartesian product of the individual sensor arrays (so that N = N_F * N_mu * N_nu).
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

        F, mu, nu = input_functions.T
        
        load_dir = self.problem_setup.lower()
        if load_dir == 'x':
            d = 0
        elif load_dir == 'y':
            d = 1
        elif load_dir == 'z':
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
        delta = delta.reshape(1,1,1,1,3)

        r_inv = 1 / r_b
        r_inv3 = 1 / (r_b ** 3)
        
        term1 = (factor[..., None] * delta) * r_inv[..., None]
        
        coord_d = coords_b[..., d:d+1]  # shape: (1, n_x, n_y, n_z, 1)
        term2 = (coords_b * coord_d) * r_inv3[..., None]
        
        u = const[..., None] * (term1 + term2)
        
        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return u, duration


    
    def produce_samples(self, filename):
        
        input_functions = self._get_input_functions()
        F, mu, nu = input_functions
        coordinates = self._get_coordinates()
        x_field, y_field, z_field = coordinates
        sensors = np.meshgrid(*input_functions, indexing="ij")
        input_functions_meshgrid = np.column_stack([i.flatten() for i in sensors])
        print(input_functions_meshgrid.shape)
        displacements, duration = self._influencefunc(input_functions_meshgrid, x_field, y_field, z_field)


        logger.info(f"Runtime for computing Kelvin solution: {duration:.2f} s")
        logger.info(f"\nData shapes:")
        logger.info(f"   Input functions meshgrid (F, mu, nu): {input_functions_meshgrid.shape}")
        logger.info(f"   Displacements u: {displacements.shape}")
        logger.info(f"   x: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}")
        logger.info(f"\nLoad magnitude min = {input_functions_meshgrid[:, 0].min():.3f}, max = {input_functions_meshgrid[:, 0].max():.3f}")
        logger.info(f"x: min = {x_field.min():3f}, max = {x_field.max():.3f}")
        logger.info(f"y: min = {y_field.min():3f}, max = {y_field.max():.3f}")
        logger.info(f"z: min = {z_field.min():3f}, max = {z_field.max():.3f}")

        np.savez(filename, F=F, mu=mu, nu=nu, x=x_field, y=y_field, z=z_field, g_u=displacements)
        logger.info(f"Saved data at {filename}")