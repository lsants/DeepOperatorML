import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')
from influence_function_kernel import kernel

def get_kernels(mesh, material, load, point, n_samples):
    kernels = np.zeros((n_samples, len(mesh)))
    print(mesh)
    for i in range(n_samples):
        m = material[0][i], material[1][i], material[2][i]
        l = load[0][i], load[1][i], load[2][i], load[3][i]
        p = point[0][i], point[1][i]
        # print(m, l, p)
        kernels[i] = kernel(mesh, m, l, p)
    return kernels

# ------- All SI units ----------
# Dataset size
N = 10

# Material data (E, ν, ρ)
E_mean, E_std = 360e6, 36e6 # Soil is ~360 MPa, Concrete is 30 GPa (took 10% std). There is a very wide range, so I'm just doing soil for now and adding concrete later
ν_min, ν_max = 0.2, 0.5
ρ_mean, ρ_std = 2e3, 5e2 
E = np.random.normal(E_mean, E_std, N)
ν = np.random.uniform(ν_min, ν_max, N)
ρ = np.random.normal(ρ_mean, ρ_std, N) # Normal distribution centered around soil density (around 2e3 kg/m3)

m_params = (E,ν,ρ)

# Load data (p0, s1, s2, ω)
ρ_steel = 7.85e3
h = 78 # Example tower in Amanda Oliveira et al.
g = 9.81
p_0 = ρ_steel*g*h*np.ones(N)
s_1 = 0*np.ones(N)
s_2 = 12.5*np.ones(N)
ω = np.random.uniform(0, 100, N)

l_params = (p_0, s_1, s_2, ω)

# Point data (r, z)
r_0, z_0 = 0.1, 0.1
d = 20
r = np.linspace(r_0, d, N)
z = np.linspace(z_0, d, N)

p = (r,z)

# ζ mesh
start, end = 0, 2
n_mesh = 300
ζ = np.linspace(start, end, n_mesh)

u_star = get_kernels(ζ, m_params, l_params, p, N)

print(u_star)


for i in range(len(u_star)):
    plt.plot(ζ, np.abs(u_star[i]))

plt.yscale('log')
plt.show()