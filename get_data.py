import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from data_generation.influence import influence
from modules.plotting import plot_label_contours, plot_label_axis

## ------------- Get parameters --------------
with open('data_generation_params.yaml') as file:
    p  =  yaml.safe_load(file)

filename = p['SAVED_DATA_PATH']
seed = p['seed']
non_dim = p['non_dim']
np.random.seed(seed)

print(filename)

# ------- Material --------
n = p['n_r']
m = p['n_z']
Es = eval(p['E'])
vs = p['poisson']
e1 = Es/(1+vs)/(1-2*vs)
c11 = e1*(1-vs)
c12 = e1*vs
c13 = e1*vs
c33 = e1*(1-vs)
c44 = e1*(1-2*vs)/2
damp = p['damp']
dens =  p['dens']

# ----------- Load ------------
z_source = p['z_source']
r_source = p['r_source']
l_source = p['l_source']
loadmag = p['load']
num_freqs = p['n_freq']
f_min, f_max = p['freq_min'], p['freq_max']
if isinstance(f_min, str) or isinstance(f_max, str):
    f_min, f_max = eval(f_min), eval(f_max)
freqs  =  f_min + np.random.rand(num_freqs) * (f_max - f_min)
q0 = loadmag / (np.pi*r_source**2)
scaling_factor = (r_source/c44)

# ----------- Problem -----------
bvptype = p['bvptype']
loadtype = p['loadtype']
component = p['component']

# ------------ Mesh --------------
r_max = p['r_max']
z_max = p['z_max']
r_min = eval(p['r_min'])*r_source
z_min = p['z_min']
r_field = np.linspace(r_min, r_max, n)
z_field = np.linspace(z_min, z_max, m)

wd = np.zeros((num_freqs, n, m), dtype=complex)

# Modify if computing dimensionless displacement
if non_dim:
    scaling_factor = 1

freqs_norm = freqs*r_source*np.sqrt(dens/c44)
c11 = c11/c44
c12 = c12/c44
c13 = c13/c44
c33 = c33/c44
c44 = c44/c44
dens = dens/dens
r_source = r_source / p['r_source']
z_source = z_source / p['r_source']
r_field_norm = r_field / p['r_source']
z_field_norm = z_field / p['r_source']

print('load', q0)
print('scaler', scaling_factor)

## -------------- Calling function ----------------
for i in tqdm(range(len(freqs_norm)), colour='Green'):
    for j in range(len(r_field_norm)):
        for k in range(len(z_field_norm)):
            wd[i, j, k] = scaling_factor * q0 * influence(
                            c11, c12, c13, c33, c44,
                            dens, damp,
                            r_field_norm[j], z_field_norm[k],
                            z_source, r_source, l_source,
                            freqs_norm[i],
                            bvptype, loadtype, component
                        )

try:
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
except FileExistsError as e:
    print('Rewriting previous data file...')

R, Z = r_field, z_field

if non_dim:
    R, Z = R / p['r_source'], Z / p['r_source']
    freqs = freqs_norm

np.savez(filename, freqs=freqs, r_field=R, z_field=Z, wd=wd)