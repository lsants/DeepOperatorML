import time
import logging
import numpy as np
from tqdm.auto import tqdm
from .data_generation_base import Datagen

logger = logging.getLogger(__name__)

class KelvinsProblem(Datagen):
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
        input_functions = np.column_stack((F_samples, 
                                       np.full(N, mu),
                                       np.full(N, nu)))
        return input_functions
    
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
        """Compute the Kelvin solution in Cartesian coordinates.
           For each branch sample (F, mu, nu) and for each grid point (x, y, z), compute the displacement vector
           using Kelvin's formula:

           Let r = sqrt(x^2 + y^2 + z^2), then
           u_i = (F / (16 * pi * mu * (1 - nu))) * (xi * z / r^3)
           u_j = (F / (16 * pi * mu * (1 - nu))) * (xj * z / r^3)
           u_k = (F / (16 * pi * mu * (1 - nu))) * ((3 - 4 * nu) / r + z^2 / r^3)

           Output is array of shape (N, n_x, n_y, n_z, 3)

        Args:
            input_functions (_type_): _description_
            x_field (array): x coordinates.
            y_field (array): y coordinates.
            z_field (array): z coordinates.
        """
        N = input_functions.shape[0]
        n_x = len(x_field)
        n_y = len(y_field)
        n_z = len(z_field)
        u = np.zeros((N, n_x, n_y, n_z, 3), dtype=np.float64)
        load_dir = self.problem_setup.lower()
        if load_dir == 'x':
            d = 0
        elif load_dir == 'y':
            d = 1
        elif load_dir == 'z':
            d = 2
        else:
            raise ValueError("Invalid load direction. Must be 'x', 'y 'or 'z'.")
        
        start = time.perf_counter_ns()
        for i in tqdm(range(N), desc="Computing Kelvin solution (Cartesian)", colour="Green"):
            F, mu, nu, = input_functions[i]
            const = F / (16 * np.pi * mu * (1 - nu))
            for ix in range(n_x):
                for iy in range(n_y):
                    for iz in range(n_z):
                        x_val = x_field[ix]
                        y_val = y_field[iy]
                        z_val = z_field[iz]
                        coord = np.array([x_val, y_val, z_val])
                        r_val = np.linalg.norm(coord)
                        if r_val < 1e-12:
                            u[i, ix, iy, iz, :] = 0.0
                        else:
                            for comp in range(3):
                                delta = 1 if comp == d else 0 # Kronecker
                                u[i, ix, iy, iz, comp] = const * (
                                    (3 - 4 * nu) * delta / r_val + (coord[comp] * coord[d] / (r_val ** 3))
                                )
        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return u, duration
    
    def produce_samples(self, filename):
        
        input_functions = self._get_input_functions()
        coordinates = self._get_coordinates()
        x_field, y_field, z_field = coordinates
        displacements, duration = self._influencefunc(input_functions, x_field, y_field, z_field)

        logger.info(f"Runtime for computing Kelvin solution: {duration:.2f} s")
        logger.info(f"\nData shapes:")
        logger.info(f"   Branch (input params): {input_functions.shape}")
        logger.info(f"   Displacements u: {displacements.shape}")
        logger.info(f"   x: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}")
        logger.info(f"\nLoad magnitude min = {input_functions[:, 0].min():.3f}, max = {input_functions[:, 0].max():.3f}")
        logger.info(f"x: min = {x_field.min():3f}, max = {x_field.max():.3f}")
        logger.info(f"y: min = {y_field.min():3f}, max = {y_field.max():.3f}")
        logger.info(f"z: min = {z_field.min():3f}, max = {z_field.max():.3f}")

        np.savez(filename, xb=input_functions, x=x_field, y=y_field, z=z_field, g_u=displacements)
        logger.info(f"Saved data at {filename}")