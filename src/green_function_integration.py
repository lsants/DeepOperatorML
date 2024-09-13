# --------------- Modules ---------------
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy import integrate
from datetime import datetime
from numerical_methods import trapezoid_rule
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

# ---------------- Paths -------------------
path_to_data = os.path.join(project_dir, 'data')
path_to_images = os.path.join(project_dir, 'images')
date = datetime.today().strftime('%Y%m%d')

# --------------- Number of points in mesh ---------------
start = 0  # starting from zero for improper integral
end = 10 # going to infinity
N = 1000 # Sensors for plot
epsilon = 1e-10  # (to avoid division by zero)

# --------------- Properties and problem scope ---------------
#   Young modulus (E), Poisson's ratio (ν) and soil density (ρ)
E = 3e6  # [Pa]
ν = 0.25
ρ = 2e3  # [Kg/m^3]

# Application of an uniformly distributed load (circular surface area: s_1 = 0) on a point (r,z).
p_0 = 1e6  # [N]
ω = 3e0  # [Hz]
s_1 = 0  # [m]
s_2 = 12.5  # [m]
a = s_2 - s_1  # Load radius [m]

# Point for which we're calculating vertical displacement u_z
r, z = (1e-1, 1e-1)

# Constants for ISOTROPIC material (Barros Thesis, 2.7)
c_11 = E*(1-ν)/((1+ν)*(1-2*ν))
c_12 = E*ν/((1+ν)*(1-2*ν))
c_33 = c_11
c_13 = c_12
c_44 = (0.5)*(c_11 - c_12)

# --------------- Coefficients in Hankel transformed variable ---------------
α = (c_33/c_44)
β = (c_11/c_44)
κ = (c_13 + c_44)/(c_44)
δ = (ρ*a**2/c_44)*ω**2  # Normalized frequency m^-1
γ = 1 + α*β - κ**2

# Define the integrand function
def kernel(ζ):
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

    kernel = -(a_7*A*np.exp(-δ*ξ_1*z) + a_8*C*np.exp(-δ*ξ_2*z))
    
    return kernel

# ---------- Integration ----------------
ζ = np.linspace(start, end, N)
y = kernel(ζ)
l_bound, u_bound = 0, np.inf

start_time_real = time.perf_counter()
result_real, error_real = integrate.quad(lambda x: np.real(kernel(x)) * x, l_bound, u_bound)
end_time_real = time.perf_counter()

start_time_imag = time.perf_counter()
result_imag, error_imag = integrate.quad(lambda x: np.imag(kernel(x)) * x, l_bound, u_bound)
end_time_imag = time.perf_counter()

runtime_real = end_time_real - start_time_real
runtime_imag = end_time_imag - start_time_imag

result = result_real
result = result_real + 1j * result_imag
error = error_real + 1j * error_imag

print(f"Integral result: {result}")
print(f"Estimated error: {error}")
print(f"Runtime: {runtime_real:.3f} seconds for real part and {runtime_imag:.3f} seconds for the imaginary part.")

# --------------------------- Plots ------------------------------
l_map = {
    'zeta': f"$\zeta$",
    'uz*': f"$|u_z^*|$"
}

plt.figure(figsize=(6, 4))
plt.plot(ζ, np.abs(y))
plt.title(l_map['uz*'], fontsize=14)
plt.xlabel(l_map['zeta'], fontsize=12)
# plt.yscale('log')
plt.tight_layout()
plt.show()
