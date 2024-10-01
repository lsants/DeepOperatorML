import time
import numpy as np
import scipy.special as sc
from scipy.integrate import quad_vec
from tqdm.auto import tqdm
from .generate_data_base import Datagen

class InfluenceFunction(Datagen):
    class IntegrandWrapper:
        def __init__(self, kernel_func, instance, points, constants, component, desc=''):
            self.kernel_func = kernel_func
            self.instance = instance
            self.points = points
            self.call_count = 0
            self.constants = constants
            self.component = component
            self.progress_bar = tqdm(
                total=None,
                leave=False,
                colour='blue',
                unit='call'
            )

        def __call__(self, ζ):
            self.call_count += 1
            self.progress_bar.update(1)
            u_star = self.kernel_func(ζ, self.instance, self.points, self.constants, self.component)
            result = ζ * u_star
            return result

        def close(self):
            self.progress_bar.close()

    def __init__(self, data_size, material_params, load_params, points_range, component, const):
        super().__init__(data_size, material_params, load_params, points_range)
        self.component = component
        self.constants = const

    def _gen_branch_data(self, data_size, material_params, load_params):
        N, _ = data_size
        E_min, E_max, nu_min, nu_max, rho_min, rho_max = material_params
        _, _, _, _, _, omega_max, omega_min = load_params
        E = np.random.uniform(E_min, E_max, N)
        ν = np.random.uniform(nu_min, nu_max, N)
        ρ = np.random.uniform(rho_min, rho_max, N)
        ω = np.random.uniform(omega_min, omega_max, N)
        branch_features = np.asarray((E, ν, ρ, ω)).T

        return branch_features
    
    def _gen_trunk_data(self, data_size, points_range):
        _, Q = data_size
        r_min, r_max, z_min, z_max = points_range
        r = np.random.uniform(r_min, r_max, Q)
        z = np.random.uniform(z_min, z_max, Q)
        trunk_features = np.asarray((r,z)).T

        return trunk_features
    
    def _kernel(self, ζ, instance_params, points, consts, var):
        ζ = np.complex128(ζ)
        
        E, ν, ρ, ω = instance_params
        ρ_steel, g, h, s_1, s_2 = consts
        p_0 = ρ_steel * g * h
        r = points[:, 0]
        z = points[:, 1]
        epsilon = 1e-10

        a = s_2 - s_1
        c_11 = E * (1 - ν) / ((1 + ν) * (1 - 2 * ν))
        c_12 = E * ν / ((1 + ν) * (1 - 2 * ν))
        c_33 = c_11
        c_13 = c_12
        c_44 = 0.5 * (c_11 - c_12)

        α = c_33 / c_44
        β = c_11 / c_44
        κ = (c_13 + c_44) / c_44
        δ = (ρ * a**2 / c_44) * ω**2
        γ = 1 + α * β - κ**2
        Φ = (γ * ζ**2 - 1 - α)**2 - 4 * α * (β * ζ**4 - β * ζ**2 - ζ**2 + 1)
        sqrt_Φ = np.sqrt(Φ)

        ξ_1 = (1 / np.sqrt(2 * α)) * np.sqrt(γ * ζ**2 - 1 - α + sqrt_Φ)
        ξ_2 = (1 / np.sqrt(2 * α)) * np.sqrt(γ * ζ**2 - 1 - α - sqrt_Φ)
        υ_1 = (α * ξ_1**2 - ζ**2 + 1) / (κ * ζ**2 + epsilon)
        υ_2 = (α * ξ_2**2 - ζ**2 + 1) / (κ * ζ**2 + epsilon)

        δζr = δ * ζ * r 
        jv0_δζr = sc.jv(0, δζr)
        jv1_δζr = sc.jv(1, δζr)

        H_0 = ((s_2 * sc.jv(1, ζ * s_2) - s_1 * sc.jv(1, ζ * s_1)) * p_0) / (ζ + epsilon)

        a_1 = υ_1 * δ * ξ_1 * (-δ * ζ * jv1_δζr)
        a_2 = υ_2 * δ * ξ_2 * (-δ * ζ * jv1_δζr)
        a_7 = δ * ξ_1 * jv0_δζr
        a_8 = δ * ξ_2 * jv0_δζr
        b_21 = (α * δ**2 * ξ_1**2 - (κ - 1) * δ**2 * ζ**2 * υ_1) * jv0_δζr
        b_22 = (α * δ**2 * ξ_2**2 - (κ - 1) * δ**2 * ζ**2 * υ_2) * jv0_δζr
        b_51 = (1 + υ_1) * δ * ξ_1 * (-δ * ζ * jv1_δζr)
        b_52 = (1 + υ_2) * δ * ξ_2 * (-δ * ζ * jv1_δζr)

        denominator = b_21 * b_52 - b_51 * b_22 + epsilon
        A = (b_52 / denominator) * (H_0 / c_44)
        C = -(b_51 / denominator) * (H_0 / c_44)

        exp_term1 = np.exp(-δ * ξ_1 * z)
        exp_term2 = np.exp(-δ * ξ_2 * z)

        if var == 'z':
            kernel = -(a_7 * A * exp_term1 + a_8 * C * exp_term2)
        else:
            kernel = a_1 * A * exp_term1 + a_2 * C * exp_term2

        return kernel
    
    def _integration(self, branch_vars, trunk_vars, l_bound=0, u_bound=np.inf):
        consts = self.constants
        var = self.component
        n = len(branch_vars)
        q = len(trunk_vars)

        integrals = np.zeros((n, q), dtype=complex)
        errors = np.zeros((n, q))
        durations = np.zeros(n)
        infos = {}

        for i in tqdm(range(n), desc='Integrating over samples', colour='green'):
            instance = branch_vars[i]

            integrand = InfluenceFunction.IntegrandWrapper(
                self._kernel,
                instance,
                trunk_vars,
                consts,
                var,
                desc=f'Integrand sample {i+1}/{n}')

            start = time.perf_counter_ns()
            integral, error, info = quad_vec(
                integrand,
                l_bound,
                u_bound,
                epsabs=1e-3,
                epsrel=1e-3,
                norm='max',
                full_output=True
            )

            integrand.close()

            end = time.perf_counter_ns()
            duration = (end - start) / 1e9

            integrals[i, :] = integral
            errors[i, :] = error
            durations[i] = duration
            infos[i] = info

        return integrals, errors, durations, infos