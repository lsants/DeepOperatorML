import os
import yaml
import numpy as np
from tqdm.auto import tqdm
from data_generation.influence import influence
from modules.plotting import plot_label_contours, plot_label_axis

## ------------- Get parameters --------------
with open('data_generation_params.yaml') as file:
    p  =  yaml.safe_load(file)

filename = p['SAVED_DATA_PATH']
seed = p['seed']
np.random.seed(seed)

print(filename)

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
loadmag = p['load']
num_freqs = p['n_freq']
f_min, f_max = p['freq_min'], p['freq_max']
if isinstance(f_min, str) or isinstance(f_max, str):
    f_min, f_max = eval(f_min), eval(f_max)
freqs  =  f_min + np.random.rand(num_freqs, ) * (f_max - f_min)
r_max = p['r_max']
z_max = p['z_max']
z_source = p['z_source']
r_source = p['r_source']
l_source = p['l_source']
bvptype = p['bvptype']
loadtype = p['loadtype']
component = p['component']
r_min = eval(p['r_min'])*r_source
z_min = p['z_min']
load_pressure = loadmag/(np.pi*r_source**2)

# Defining mesh
r_field = np.linspace(r_min, r_max, n)
z_field = np.linspace(z_min, z_max, m)
wd = np.zeros((num_freqs, n, m), dtype=complex)
wd_normalized = wd

'''Normalization (we normalize the material constants, frequency, load radius and mesh
in order to compare results with Rajapakse & Wang (1993)):
'''
c11_normalized = c11/c44
c12_normalized = c12/c44
c13_normalized = c13/c44
c33_normalized = c33/c44
c44_normalized = c44/c44
dens_normalized = dens/dens
freqs_normalized = freqs*r_source*np.sqrt(dens/c44)
r_source_normalized = r_source/r_source
z_source_normalized = z_source/r_source
r,z = r_field, z_field
r_normalized = r_field / r_source
z_normalized = z_field / r_source

scaling_factor = (load_pressure*r_source)/c44

## -------------- Calling function ----------------
for i in tqdm(range(len(freqs_normalized)), colour='Green'):
    for j in range(len(r_normalized)):
        for k in range(len(z_normalized)):
            wd_normalized[i, j, k] = influence(
                            c11_normalized, c12_normalized, c13_normalized, c33_normalized, c44_normalized,
                            dens_normalized, damp,
                            r_normalized[j], z_normalized[k],
                            z_source_normalized, r_source_normalized, l_source,
                            freqs_normalized[i],
                            bvptype, loadtype, component
                        )
wd = wd_normalized * scaling_factor

try:
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
except FileExistsError as e:
    print('Rewriting previous data file...')

# np.savez(filename, freqs=freqs, r_field=r_field, z_field=z_field, wd=wd)

## ----------- Plot -------------
R, Z = r_normalized, z_normalized
f_index = 0

wd_plot = wd_normalized[f_index]

fig = plot_label_contours(R,Z,wd_plot, freqs_normalized[f_index], full=True)
# plot_label_axis(R, Z, wd_plot, freqs_normalized[f_index], axis=p['axis_plot'])
