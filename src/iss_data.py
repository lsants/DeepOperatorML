import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')
from influence_function_kernel import kernel

def get_kernels(mesh, material, load, point, n_samples):
    kernels = np.zeros((n_samples, len(mesh)))
    for i in range(n_samples):
        m = material[0][i], material[1][i], material[2][i]
        l = load[0][i]
        p = point[0][i], point[1][i]
        for j,e in enumerate(mesh):
            kernels[i][j] = kernel(e, m, l, p)
    return kernels

def integrate_kernels(material, load, point, n_samples, lower_bound, upper_bound):
    integrals = []
    errors = []
    durations = []
    for i in range(n_samples):
        m = material[0][i], material[1][i], material[2][i]
        l = load[0][i]
        p = point[0][i], point[1][i]
        start = time.perf_counter_ns()
        integral, error = integrate.quad(lambda x: x*kernel(x, m, l, p), lower_bound, upper_bound, complex_func=True)
        end = time.perf_counter_ns()
        duration = (end - start)/1e6
        # print(integral, error)
        # print(f"Integration took: {duration:.2f} ms")
        integrals.append(integral)
        errors.append(error)
        durations.append(duration)
    return np.array(integrals), np.array(errors), np.array(durations)

np.random.seed(42)

# ------- All SI units ----------
# Dataset size
N = 15

# ---------------------------------- Material data (E, ν, ρ) ------------------------
E_mean, E_std = 360e6, 36e6 # Soil is ~360 MPa, Concrete is 30 GPa (took 10% std). There is a very wide range, so I'm just doing soil for now and adding concrete later
ν_min, ν_max = 0.2, 0.5
ρ_mean, ρ_std = 2e3, 5e2 
E = np.random.normal(E_mean, E_std, N)
ν = np.random.uniform(ν_min, ν_max, N)
ρ = np.random.normal(ρ_mean, ρ_std, N) # Normal distribution centered around soil density (around 2e3 kg/m3)
m_params = (E,ν,ρ)

# --------------------------------- Load data (in this case only ω) ------------------------
ρ_steel = 7.85e3
h = 78 # Example tower in Amanda Oliveira et al. (Fix wind turbine)
g = 9.81
p_0 = ρ_steel*g*h
s_1 = 0
s_2 = 12.5
ω = np.random.uniform(0, 100, N)
l_params = (ω,)
# ------------------------------ Point data (r, z) --------------------------------
r_0, z_0 = 0.1, 0.1
d = 20 # length of square centered at the origin where the trunk will be trained
r = np.linspace(r_0, d, N)
z = np.linspace(z_0, d, N)
p = (r,z)

# ---------------------------- Computing kernel -----------------------------------
start, end = 0, 10
n_mesh = 300
ζ = np.linspace(start, end, n_mesh)
u_star = get_kernels(ζ, m_params, l_params, p, N)

# ---------------------------- Integrating -----------------------------------------
l_bound = start
u_bound = np.inf
integrals, errors, durations = integrate_kernels(m_params, l_params, p, N, l_bound, u_bound)

print('-----------------------------------------------------------------------')
print(f"Runtime for integration: {durations.mean():.2f} ±  {durations.std():.2f} ms")
print('-----------------------------------------------------------------------')

# ---------------------------- Processing data for training ------------------------
features = np.asarray(m_params + l_params + p)
labels = integrals

# ------------------------------ Plots ---------------------------------------------
ω_min = ω[np.argmin(ω)]
print(f'ω min. = {ω_min:.3f} Hz')
print(f'p_0 = {p_0:.3E} N')

plt.plot(ζ, np.abs(u_star[np.argmin(ω)]))
plt.xlabel(r"$ζ$")
plt.ylabel(r"$u^*_{zz}$")
plt.ylim([-1e-3, 1e1])
plt.tight_layout()
plt.show()

# ----------------------------- Saving data -------------------------------------
# --- Save dataset ---
if __name__ == '__main__':
    np.savez(os.path.join(path_to_data, 'iss_dataset.npz'), X=features, y=labels)