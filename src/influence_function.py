import time
import numpy as np
import scipy.special as sc
import numpy.typing as npt
from scipy.integrate import quad_vec
from tqdm.auto import tqdm

''' For kernel: material parameters, load parameters (geometry, position, frequency and magnitude), point
    Material parameters: (E, ν, ρ)
    Load parameters: (p0, s1, s2, ω)
    Point: (r,z)
'''


class IntegrandWrapper:
    def __init__(self, kernel_func, instance, points, desc=''):
        self.kernel_func = kernel_func
        self.instance = instance
        self.points = points
        self.call_count = 0
        self.progress_bar = tqdm(
            total=None,
            leave=False,
            colour='blue',
            unit='call'
        )

    def __call__(self, ζ):
        self.call_count += 1
        self.progress_bar.update(1)
        result = ζ * np.array([self.kernel_func(ζ, self.instance, p) for p in self.points])

        return result

    def close(self):
        self.progress_bar.close()

def kernel_z(ζ, params: tuple, point:tuple) -> npt.NDArray:
    """Generate influence function kernel in r based on mesh and parameters

    Args:
        ζ: Scaled Hankel space variable.
        material_params (tuple): tuple consisting of Young modulus, Poisson's ratio density of the medium and load frequency. 
        point (tuple): coordinates r and z where the influence function shall be evaluated

    Returns:
        npt.NDArray[np.complex128]: Kernel for influence function in the z direction. Will be used for integration.
    """
    ζ = np.asarray(ζ, dtype=np.complex128)
    
    # Parameters
    E, ν, ρ, ω = params
    ρ_steel = 7.85e3
    h = 78 # Example tower in Amanda Oliveira et al.
    g = 9.81
    p_0 = ρ_steel*g*h
    s_1 = 0
    s_2 = 12.5
    r,z = point
    epsilon = 1e-10

    a = s_2 - s_1
    c_11 = E*(1-ν)/((1+ν)*(1-2*ν))
    c_12 = E*ν/((1+ν)*(1-2*ν))
    c_33 = c_11
    c_13 = c_12
    c_44 = (0.5)*(c_11 - c_12)

    α = (c_33/c_44)
    β = (c_11/c_44)
    κ = (c_13 + c_44)/(c_44)
    δ = (ρ*a**2/c_44)*ω**2
    γ = 1 + α*β - κ**2
    Φ = (γ*ζ**2 - 1 - α)**2 - 4*α*(β*ζ**4 - β*ζ**2 - ζ**2 + 1)
    ξ_1 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α + np.sqrt(Φ))
    ξ_2 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α - np.sqrt(Φ))
    υ_1 = (α*ξ_1**2 - ζ**2 + 1)/(κ*ζ**2 + epsilon)
    υ_2 = (α*ξ_2**2 - ζ**2 + 1)/(κ*ζ**2 + epsilon)

    H_0 = ((s_2*sc.jv(1, ζ*s_2) - s_1*sc.jv(1, ζ*s_1))*p_0)/(ζ + epsilon)
    a_7 = δ*ξ_1*sc.jv(0, δ*ζ*r)
    a_8 = δ*ξ_2*sc.jv(0, δ*ζ*r)
    b_21 = (α*δ**2*ξ_1**2 - (κ-1)*δ**2*ζ**2*υ_1)*(sc.jv(0, δ*ζ*r))
    b_22 = (α*δ**2*ξ_2**2 - (κ-1)*δ**2*ζ**2*υ_2)*(sc.jv(0, δ*ζ*r))
    b_51 = (1 + υ_1)*δ*ξ_1*(-δ*ζ*sc.jv(1, δ*ζ*r))
    b_52 = (1 + υ_2)*δ*ξ_2*(-δ*ζ*sc.jv(1, δ*ζ*r))

    denominator = b_21*b_52 - b_51*b_22 + epsilon
    A = (b_52/(denominator)) * (H_0/c_44)
    C = -(b_51/(denominator)) * (H_0/c_44)

    kernel = -(a_7*A*np.exp(-δ*ξ_1*z) + a_8*C*np.exp(-δ*ξ_2*z))
    
    return kernel

def kernel_r(ζ, params: tuple, point:tuple) -> npt.NDArray:
    """Generate influence function kernel in r based on mesh and parameters

    Args:
        ζ: Scaled Hankel space variable.
        material_params (tuple): tuple consisting of Young modulus, Poisson's ratio density of the medium and load frequency.
        point (tuple): coordinates r and z where the influence function shall be evaluated

    Returns:
        npt.NDArray[np.complex128]: Kernel for influence function in the r direction. Will be used for integration.
    """
    ζ = np.asarray(ζ, dtype=np.complex128)
    
    # Parameters
    E, ν, ρ, ω = params
    ρ_steel = 7.85e3
    h = 78 # Example tower in Amanda Oliveira et al.
    g = 9.81
    p_0 = ρ_steel*g*h
    s_1 = 0
    s_2 = 12.5
    r,z = point
    epsilon = 1e-10

    a = s_2 - s_1
    c_11 = E*(1-ν)/((1+ν)*(1-2*ν))
    c_12 = E*ν/((1+ν)*(1-2*ν))
    c_33 = c_11
    c_13 = c_12
    c_44 = (0.5)*(c_11 - c_12)

    α = (c_33/c_44)
    β = (c_11/c_44)
    κ = (c_13 + c_44)/(c_44)
    δ = (ρ*a**2/c_44)*ω**2
    γ = 1 + α*β - κ**2
    Φ = (γ*ζ**2 - 1 - α)**2 - 4*α*(β*ζ**4 - β*ζ**2 - ζ**2 + 1)
    ξ_1 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α + np.sqrt(Φ))
    ξ_2 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α - np.sqrt(Φ))
    υ_1 = (α*ξ_1**2 - ζ**2 + 1)/(κ*ζ**2 + epsilon)
    υ_2 = (α*ξ_2**2 - ζ**2 + 1)/(κ*ζ**2 + epsilon)

    H_0 = ((s_2*sc.jv(1, ζ*s_2) - s_1*sc.jv(1, ζ*s_1))*p_0)/(ζ + epsilon)
    a_1 = υ_1*δ*ξ_1*(-δ*ζ*sc.jv(1, δ*ζ*r))
    a_2 = υ_2*δ*ξ_2*(-δ*ζ*sc.jv(1, δ*ζ*r))
    b_21 = (α*δ**2*ξ_1**2 - (κ-1)*δ**2*ζ**2*υ_1)*(sc.jv(0, δ*ζ*r))
    b_22 = (α*δ**2*ξ_2**2 - (κ-1)*δ**2*ζ**2*υ_2)*(sc.jv(0, δ*ζ*r))
    b_51 = (1 + υ_1)*δ*ξ_1*(-δ*ζ*sc.jv(1, δ*ζ*r))
    b_52 = (1 + υ_2)*δ*ξ_2*(-δ*ζ*sc.jv(1, δ*ζ*r))

    denominator = b_21*b_52 - b_51*b_22 + epsilon
    A = (b_52/(denominator)) * (H_0/c_44)
    C = -(b_51/(denominator)) * (H_0/c_44)

    kernel = a_1*A*np.exp(-δ*ξ_1*z) + a_2*C*np.exp(-δ*ξ_2*z)
    
    return kernel

def get_kernels(mesh, params, points):
    E, ν, ρ, ω = params
    r_points, z_points = points

    n_samples = len(E)
    n_points = len(r_points)
    n_mesh = len(mesh)
    kernels_r = np.zeros((n_samples, n_points, n_mesh), dtype=np.complex128)
    kernels_z = np.zeros((n_samples, n_points, n_mesh), dtype=np.complex128)

    for i in tqdm(range(n_samples), colour='GREEN'):
        instance = (E[i], ν[i], ρ[i], ω[i])
        for j in range(n_points):
            p = (r_points[j], z_points[j])
            kernels_r[i, j, :] = kernel_r(mesh, instance, p)
            kernels_z[i, j, :] = kernel_z(mesh, instance, p)

    return kernels_r, kernels_z


def integrate_kernels(branch_vars, trunk_vars, lower_bound, upper_bound):
    n = len(branch_vars)
    q = len(trunk_vars)
    E, ν, ρ, ω = branch_vars.T
    points = trunk_vars

    integrals_r = np.zeros((n, q), dtype=complex)
    integrals_z = np.zeros((n, q), dtype=complex)
    errors_r = np.zeros((n, q))
    errors_z = np.zeros((n, q))
    durations = np.zeros(n)

    for i in tqdm(range(n), desc='Integrating over samples', colour='green'):
        instance = (E[i], ν[i], ρ[i], ω[i])

        integrand_r = IntegrandWrapper(kernel_r, instance, points, desc=f'Integrand_r sample {i+1}/{n}')
        integrand_z = IntegrandWrapper(kernel_z, instance, points, desc=f'Integrand_z sample {i+1}/{n}')

        start = time.perf_counter_ns()
        integral_r, error_r = quad_vec(
            integrand_r,
            lower_bound,
            upper_bound,
            epsabs=1e-1,
            epsrel=1e-1,
            norm='max'
        )
        integral_z, error_z = quad_vec(
            integrand_z,
            lower_bound,
            upper_bound,
            epsabs=1e-1,
            epsrel=1e-1,
            norm='max'
        )

        integrand_r.close()
        integrand_z.close()

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9

        integrals_r[i, :] = integral_r
        integrals_z[i, :] = integral_z
        errors_r[i, :] = error_r
        errors_z[i, :] = error_z
        durations[i] = duration

    return integrals_r, integrals_z, errors_r, errors_z, durations