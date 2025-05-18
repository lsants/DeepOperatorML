import time
import logging
import numpy as np
from tqdm.auto import tqdm
from ..base_generator import Datagen
from ..rajapakse_fixed_material.influence import influence

logger = logging.getLogger(__name__)

class DynamicHomogeneousMaterialProblem(Datagen):
    """
    Data generation for a PDE with homogeneous isotropic material,
    where each sample has distinct scalar parameters E, nu, rho, and omega.
    """

    def __init__(self, data_size, material_params, load_params, mesh_params, problem_setup):
        """
        Args:
            data_size (tuple): (N, nr, nz)
                - N: number of PDE samples
                - nr: number of radial points for the solution field
                - nz: number of vertical points

            material_params (tuple): (E_min, E_max, nu_min, nu_max, dens_min, dens_max, damp)
                - E_min, E_max: min and max for Young's modulus
                - nu_min, nu_max: min and max for Poisson ratio
                - dens_min, dens_max: min and max for density
                - damp: damping factor (constant for all samples, or you can sample if needed)

            load_params (tuple): (omega_min, omega_max, z_source, l_source, r_source)
                - omega_min, omega_max: freq range
                - z_source, l_source, r_source: geometry coords for the PDE

            mesh_params (tuple): (r_min, r_max, z_min, z_max)
                - domain extents for radial and vertical directions

            problem_setup (tuple): (component, loadtype, bvptype)
                - PDE options used by `influence`
        """
        super().__init__(data_size, material_params, load_params, mesh_params, problem_setup)

    def _get_input_functions(self):
        """
        1) Sample the scalar parameters E, nu, rho, omega for each sample i in [0..N-1].
        2) Compute c44 and delta as auxiliary quantities if needed for your PDE.
        """
        N, _, _ = self.data_size

        # Unpack load_params and material_params
        omega_min, omega_max, z_source, l_source, r_source = self.load_params
        E_min, E_max, nu_min, nu_max, dens_min, dens_max, damp = self.material_params

        # -- 1) Sample E with log-uniform, and nu, rho, omega with linear uniform
        logE_min, logE_max = np.log10(E_min), np.log10(E_max)
        E_samples = 10**(np.random.uniform(logE_min, logE_max, size=N))

        nu_samples   = np.random.uniform(nu_min,   nu_max,   size=N)
        rho_samples  = np.random.uniform(dens_min, dens_max, size=N)
        omega_samples= np.random.uniform(omega_min, omega_max,size=N)

        # -- 2) Compute c44 and delta if needed by the PDE
        # e1 = E/(1+nu)/(1-2nu), c44 = e1*(1-2nu)/2
        e1 = E_samples / ((1 + nu_samples)*(1 - 2*nu_samples))
        c44_samples = e1*(1 - 2*nu_samples)/2.0

        # delta = dimensionless freq (example formula; depends on your PDE)
        delta_samples = omega_samples * r_source * np.sqrt(rho_samples / c44_samples)

        return (E_samples, nu_samples, rho_samples, damp,
                omega_samples, c44_samples, delta_samples)
    
    def _get_coordinates(self):
        """
        Build the (r, z) grids for evaluating the solution field.
        """
        _, n_r, n_z = self.data_size
        _, _, _, _, r_source = self.load_params
        r_min, r_max, z_min, z_max = self.mesh_params

        # Shift r_min to avoid singularity at r=0, then normalize by r_source
        modified_r_min = r_min + (r_source * 1e-2)
        r_field = np.linspace(modified_r_min, r_max, n_r) / r_source
        z_field = np.linspace(z_min, z_max, n_z) / r_source

        return (r_field, z_field)
    
    def _influencefunc(self, E, nu, rho, damp, omega, c44, delta, r_field, z_field):
        """
        Compute PDE solution for each sample i by calling `influence`.
        
        E[i], nu[i], rho[i], omega[i] are the scalar parameters for sample i.
        c44[i], delta[i] are derived scalars. We loop over i and fill out wd[i,:,:].
        """
        N, n_r, n_z = self.data_size
        # geometry & PDE setup
        omega_min, omega_max, z_source, l_source, r_s_temp = self.load_params
        component, loadtype, bvptype = self.problem_setup

        # Prepare output array
        wd = np.zeros((N, n_r, n_z), dtype=complex)

        # Convert dimensionless geometry if needed
        z_source = z_source / r_s_temp
        r_source = r_s_temp / r_s_temp  # => 1.0

        # Solve PDE for each of the N samples
        start = time.perf_counter_ns()

        for i in tqdm(range(N), desc="Solving PDE", colour='Green'):
            # Extract the scalar parameters for sample i
            E_i     = E[i]
            nu_i    = nu[i]
            rho_i   = rho[i]
            damp_i  = damp  # if you want to vary damp, sample it in _get_input_functions
            c44_i   = c44[i]
            delta_i = delta[i]
            omega_i = omega[i]

            # Recompute local dimensionless constants if needed
            e1_i  = E_i / ((1 + nu_i)*(1 - 2*nu_i))
            c11_i = e1_i*(1 - nu_i)
            c12_i = e1_i*nu_i
            c13_i = e1_i*nu_i
            c33_i = e1_i*(1 - nu_i)
            c44_i = e1_i*(1 - 2*nu_i)/2.0

            # Dimensionless scale
            c11_i /= c44_i
            c12_i /= c44_i
            c13_i /= c44_i
            c33_i /= c44_i
            c44_i /= c44_i  # => 1.0
            # Similarly for density, if you need dimensionless  => rho_i / reference_rho
            # Here we just set it to 1.0 in the call, if that's your PDE convention.

            # Evaluate PDE solution at each (r_field[j], z_field[k])
            for j in range(n_r):
                for k in range(n_z):
                    wd[i, j, k] = influence(
                        c11_i, c12_i, c13_i, c33_i, c44_i,
                        1.0,          # dimensionless rho
                        damp_i,
                        r_field[j], z_field[k],
                        z_source, r_source, l_source,
                        delta_i,      # or omega_i if needed
                        bvptype, loadtype, component
                    )

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return wd, duration
    
    def produce_samples(self, filename):
        """
        Main workflow to:
         1) Sample the scalar parameters E, nu, rho, omega (and derived c44, delta).
         2) Build coordinate grids (r_field, z_field).
         3) Solve PDE for each sample.
         4) Save results to .npz file.
        """
        # 1) Sample parameter arrays for N PDE scenarios
        E, nu, rho, damp, omega, c44, delta = self._get_input_functions()

        # 2) Build (r, z) coordinate grids
        r_field, z_field = self._get_coordinates()

        # 3) Solve PDE for each sample
        displacements, comp_time = self._influencefunc(
            E, nu, rho, damp, omega, c44, delta,
            r_field, z_field
        )

        # 4) Logging and saving
        logger.info(f"Runtime for PDE solutions: {comp_time:.2f} s")
        logger.info(
            f"Data shapes:\n"
            f"\tE, nu, rho, omega:\t{E.shape}, {nu.shape}, {rho.shape}, {omega.shape}\n"
            f"\tdisplacements:\t\t{displacements.shape}\n"
            f"\tr:\t{r_field.shape},\tz:\t{z_field.shape}"
        )
        logger.info(
            f"E range: [{E.min():.2e}, {E.max():.2e}] | "
            f"nu range: [{nu.min():.3f}, {nu.max():.3f}] | "
            f"rho range: [{rho.min():.1f}, {rho.max():.1f}] | "
            f"omega range: [{omega.min():.1f}, {omega.max():.1f}]"
        )
        logger.info(f"delta range: [{delta.min():.2f}, {delta.max():.2f}]")

        np.savez(
            filename,
            E=E, nu=nu, rho=rho, omega=omega,
            r=r_field, z=z_field, g_u=displacements
        )
        logger.info(f"Saved dataset to {filename}")
