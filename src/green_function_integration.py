# --------------- Modules ---------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy import integrate

# --------------- Parameters ---------------
images_path = '/home/lsantiago/workspace/ic/Relatorio/Imagens'

# --------------- Number of points in mesh ---------------
start = 1e-4  # avoid starting at zero due to division by zero error
end = 10
N = 100000
epsilon = 1e-10  # (to avoid division by zero)

# --------------- Material properties and problem scope ---------------
#   Young modulus (E), Poisson's ratio (ν) and soil density (ρ)
E = 2.5e9  # [Pa]
ν = 0.25
ρ = 1e3  # [Kg/m^3]

# Application of an uniformly distributed load (circular surface area: s_1 = 0) on a point (r,z).
# Values are arbitrary
p_0 = 1e2  # [N]
ω = np.sqrt(3)*1e3  # [Hz]
s_1 = 0  # [m]
s_2 = 1  # [m]
a = s_2 - s_1  # Load radius [m]
# [m] (point for which we're calculating the vertical displacement)
r, z = (2, 1e-3)

# Constants for ISOTROPIC material (Barros Thesis, 2.7)
c_11 = E*(1-ν)/((1+ν)*(1-2*ν))
c_12 = E*ν/((1+ν)*(1-2*ν))
c_33 = c_11
c_13 = c_12
c_44 = (0.5)*(c_11 - c_12)

# --------------- Parameters for Hankel transforms ---------------
ζ = np.linspace(start, end, N)  # Integration variable
α = (c_33/c_44)
β = (c_11/c_44)
κ = (c_13 + c_44)/(c_44)
δ = (ρ*a**2/c_44)*ω**2  # Normalized frequency m^-1
γ = 1 + α*β - κ**2
# change to complex when it is an input to the square root function to avoid error when plotting complex exponential
ζ = ζ.astype(complex)

# --------------- Φ, ξ_i, υ_1 ---------------
Φ = (γ*ζ**2 - 1 - α)**2 - 4*α*(β*ζ**4 - β*ζ**2 - ζ**2 + 1)
ξ_1 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α + np.sqrt(Φ))
ξ_2 = (1/np.sqrt(2*α))*np.sqrt(γ*ζ**2 - 1 - α - np.sqrt(Φ))
υ_1 = (α*ξ_1**2 - ζ**2 + 1)/(κ*ζ**2)
υ_2 = (α*ξ_2**2 - ζ**2 + 1)/(κ*ζ**2)

# --------------- H_0, a_i, b_i ---------------
H_0 = ((s_2*sc.jv(1, ζ*s_2) - s_1*sc.jv(1, ζ*s_1))*p_0)/(ζ + epsilon)
a_7 = δ*ξ_1*sc.jv(0, δ*ζ*r)
a_8 = δ*ξ_2*sc.jv(0, δ*ζ*r)
b_21 = (α*δ**2*ξ_1**2 - (κ-1)*δ**2*ζ**2*υ_1)*(sc.jv(0, δ*ζ*r))
b_22 = (α*δ**2*ξ_2**2 - (κ-1)*δ**2*ζ**2*υ_2)*(sc.jv(0, δ*ζ*r))
b_51 = (1 + υ_1)*δ*ξ_1*(-δ*ζ*sc.jv(1, δ*ζ*r))
b_52 = (1 + υ_2)*δ*ξ_2*(-δ*ζ*sc.jv(1, δ*ζ*r))

# --------------- Arbitrary functions from boundary conditions ---------------
denominator = b_21*b_52 - b_51*b_22 + epsilon
A = (b_52/(denominator)) * (H_0/c_44)
C = -(b_51/(denominator)) * (H_0/c_44)

# Define the integrand function
def integrand(ζ):
    # Recalculate values for the given ζ
    ζ = np.asarray(ζ, dtype=complex)
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

    u_z = -(a_7*A*np.exp(-δ*ξ_1*z) + a_8*C*np.exp(-δ*ξ_2*z))
    
    return np.abs(u_z)

# Perform adaptive Gaussian quadrature
result, error = integrate.quad(lambda x: np.real(integrand(x)), start, end)

print(f"Integral result: {result}")
print(f"Estimated error: {error}")

# --------------------------- Plots ------------------------------
# u_z plot
x = ζ
y = integrand(x)
l_map = {
    'zeta': f"$\zeta$",
    'uz*': f"$|u_z^*|$"
}

valid_indices = np.abs(y) > 0
x_valid = ζ[valid_indices]
y_valid = np.abs(y)[valid_indices]

plt.figure(figsize=(6, 4))
plt.plot(x_valid, y_valid)
plt.title(l_map['uz*'], fontsize=14)
plt.xlabel(l_map['zeta'], fontsize=12)
plt.yscale('log')
plt.tight_layout()
plt.show()
