import numpy as np
import scipy.special as sc
import numpy.typing as npt

''' For kernel: material parameters, load parameters (geometry, position, frequency and magnitude), point
    Material parameters: (E, ν, ρ)
    Load parameters: (p0, s1, s2, ω)
    Point: (r,z)
'''

def kernel_z(ζ, params: tuple, point:tuple) -> npt.NDArray:
    """Generate influence function kernel in r based on mesh and parameters

    Args:
        ζ: Scaled Hankel space variable.
        material_params (tuple): tuple consisting of Young modulus, Poisson's ratio density of the medium and load frequency.
        point (tuple): coordinates r and z where the influence function shall be evaluated

    Returns:
        npt.NDArray[np.complex128]: Kernel for influence function in the z direction. Will be used for integration.
    """
    ζ = complex(ζ)
    
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
    ζ = complex(ζ)
    
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