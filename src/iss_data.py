import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')
from influence_function_kernel import kernel_r, kernel_z

np.random.seed(42)

def get_kernels(mesh, params, point, n_samples):
    kernels_r = np.zeros((n_samples, len(mesh)))
    kernels_z = np.zeros((n_samples, len(mesh)))
    E, ν, ρ, ω  = params
    for i in range(n_samples):
        instance = (E[i], ν[i], ρ[i], ω[i])
        p = point[0][i], point[1][i]
        for j,e in enumerate(mesh):
            kernels_r[i][j] = kernel_r(e, instance, p)
            kernels_z[i][j] =  kernel_z(e, instance, p)
    return (kernels_r, kernels_r)

def integrate_kernels(branch_vars, trunk_vars, lower_bound, upper_bound):
    integrals_r = []
    integrals_z = []
    errors_r = []
    errors_z = []
    durations = []
    n = len(branch_vars)
    q = len(trunk_vars)
    E, ν, ρ, ω = branch_vars.T
    point = trunk_vars
    for i in tqdm(range(n), colour='GREEN'):
        for j in range(q):
            instance = (E[i], ν[i], ρ[i], ω[i])
            p = point[j]
            start = time.perf_counter_ns()
            integral_r, error_r = integrate.quad(lambda ζ: ζ*kernel_r(ζ, instance, p), lower_bound, upper_bound, complex_func=True)
            integral_z, error_z = integrate.quad(lambda ζ: ζ*kernel_z(ζ, instance, p), lower_bound, upper_bound, complex_func=True)
            end = time.perf_counter_ns()
            duration = (end - start)/1e6
            # print(integral_r, error_r)
            # print(f"Integration took: {duration:.2f} ms")
            integrals_r.append(integral_r)
            integrals_z.append(integral_z)
            errors_r.append(error_r)
            errors_z.append(error_z)
            durations.append(duration)
    integrals_r = np.array(integrals_r).reshape(len(branch_vars), len(trunk_vars))
    integrals_z = np.array(integrals_z).reshape(len(branch_vars), len(trunk_vars))
    return np.array(integrals_r), np.array(integrals_z), np.array(errors_r), np.array(errors_z), np.array(durations)

# ------- All SI units ----------
# Dataset size
N = 50 # Branch
q = 100 # Trunk

# ---------------------------------- Material data (E, ν, ρ) ------------------------
E_mean, E_std = 360e6, 36e6 # Soil is ~360 MPa, Concrete is 30 GPa (took 10% std). There is a very wide range, so I'm just doing soil for now and adding concrete later
ν_min, ν_max = 0.2, 0.5
ρ_mean, ρ_std = 2e3, 5e2 
E = np.random.normal(E_mean, E_std, N)
ν = np.random.uniform(ν_min, ν_max, N)
ρ = np.random.normal(ρ_mean, ρ_std, N) # Normal distribution centered around soil density (around 2e3 kg/m3)
m_params = (E,ν,ρ)

ρ_steel = 7.85e3
h = 78 # Example tower in Amanda Oliveira et al. (Fix wind turbine)
g = 9.81
p_0 = ρ_steel*g*h
s_1 = 0
s_2 = 12.5
ω_min, ω_max = 0, 100
ω = np.random.uniform(ω_min, ω_max, N)
l_params = (ω,)

# ------------------------------ Point data (r, z) --------------------------------
r_0, z_0 = 0.001, 0.001
d = 10 # length of square centered at the origin where the trunk will be trained (space domain, must define a bound)
r = np.random.uniform(r_0, d, q)
z = np.random.uniform(z_0, d, q)
p = (r,z)

params = m_params + l_params

branch_features = np.asarray(m_params + l_params).T
branch_features[:, :-1] = branch_features[0,:-1] # Fixing material for tacking simpler problem at first
trunk_features = np.asarray(p).T

# ---------------------------- Computing kernel for plot -----------------------------------
start, end = 0, 10
n_mesh = 100
ζ = np.linspace(start, end, n_mesh)
u_star_r, u_star_z = get_kernels(ζ, params, p, N)

# ---------------------------- Integrating ------------------------------------------------
l_bound = start
u_bound = np.inf
integrals_r, integrals_z, errors_r, errors_z, durations = integrate_kernels(branch_features, trunk_features, l_bound, u_bound)

print('-----------------------------------------------------------------------')
print(f"Runtime for integration: {durations.mean():.2f} ±  {durations.std():.2f} ms")
print('-----------------------------------------------------------------------')

integrals = np.concatenate((integrals_r, integrals_z), axis=1)
labels = integrals

# ------------------------------ Plots -------------------------------------------------------
# ω_test = ω[np.argmin(ω)]
# print(f'ω min. = {ω_test:.3f} Hz')
# print(f'p_0 = {p_0:.3E} N')

# plt.plot(ζ, np.abs(u_star_z[np.argmin(ω)]))
# plt.xlabel(r"$ζ$")
# plt.ylabel(r"$u^*_{zz}$")
# plt.ylim([-1e-3, 1e1])
# plt.tight_layout()
# plt.show()

#-------------------------------- Split training and test set -----------------------------
test_size = 0.2
train_rows = int(N * (1 - test_size))
test_rows = N - train_rows           

train_cols = int(q * (1 - test_size))
test_cols = q - train_cols           

u_train, u_test, labels_train, labels_test = train_test_split(branch_features, labels, test_size=test_size, random_state=42)

labels_train_r, labels_train_z = labels_train[:,:q], labels_train[:,q:]
labels_test_r, labels_test_z = labels_test[:,:q], labels_test[:,q:]

train_data = (u_train, labels_train_r, labels_train_z)
test_data = (u_test, labels_test_r, labels_test_z)

train_shapes = '\n'.join([str(i.shape) for i in train_data]) 
test_shapes = '\n'.join([str(i.shape) for i in test_data])

print(f"Train sizes (u, Gr, Gz): \n{train_shapes}, \nTest sizes (u, Gr, Gz): \n{test_shapes}")

# ----------------------------- Saving data -------------------------------------
if __name__ == '__main__':
    np.savez(os.path.join(path_to_data, 'iss_dataset_full_fixed.npz'), X_branch=branch_features, X_trunk=trunk_features, y_r=integrals_r, y_z=integrals_z)
    np.savez(os.path.join(path_to_data, 'iss_train_fixed.npz'), X_branch=u_train, X_trunk=trunk_features, y_r=labels_train_r, y_z=labels_train_z)
    np.savez(os.path.join(path_to_data, 'iss_test_fixed.npz'), X_branch=u_test, X_trunk=trunk_features, y_r=labels_test_r, y_z=labels_test_z)