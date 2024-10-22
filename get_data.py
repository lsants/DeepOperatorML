import os
import yaml
import numpy as np
from tqdm.auto import tqdm
from data_generation.influence import influence
from modules.plotting import plot_labels

## ------------- Get parameters --------------
with open('data_generation_params.yaml') as file:
    p  =  yaml.safe_load(file)

filename = p['SAVED_DATA_PATH']

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
freqs  =  p['freq_min'] + np.random.rand(num_freqs) * (p['freq_max'] - p['freq_min'])
r_min = eval(p['r_min'])
z_min = eval(p['z_min'])
r_max = p['r_max']
z_max = p['z_max']
z_source = p['z_source']
r_source = p['r_source']
l_source = p['l_source']
bvptype = p['bvptype']
loadtype = p['loadtype']
component = p['component']

# Defining mesh
r_field = np.linspace(r_min, r_max, n)
z_field = np.linspace(z_min, z_max, m)
wd = np.zeros((n,m, num_freqs), dtype=complex)

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
load_stress = loadmag/(np.pi*r_source**2)
r_normalized = r_field / r_source
z_normalized = r_field / r_source

## -------------- Calling function ----------------
for k in tqdm(range(len(freqs_normalized)), colour='Green'):
    for i in range(len(r_normalized)):
        for j in range(len(z_normalized)):
            wd[i, j, k] = load_stress*(r_source/c44)*influence(
                            c11_normalized, c12_normalized, c13_normalized, c33_normalized, c44_normalized,
                            dens_normalized, damp,
                            r_normalized[i], z_normalized[j],
                            z_source, r_source, l_source,
                            freqs_normalized[k],
                            bvptype, loadtype, component
                        )

try:
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
except FileExistsError as e:
    print('Rewriting previous data file...')

np.savez(filename, freqs=freqs, r_field=r_field, z_field=z_field, wd=wd)

## ----------- Plot ------------- EDIT PLOT UNITS
R, Z = r_field, z_field
f_index = 0
wd_transposed = wd.transpose(2,1,0)
wd_f = wd_transposed[f_index]
wd_plot = wd_f
plot_labels(R,Z,wd_plot, freqs[f_index], plot_type='abs')
