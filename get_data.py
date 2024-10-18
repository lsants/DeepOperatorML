import os
import yaml
import numpy as np
from tqdm.auto import tqdm
from data_generation.influence import influence
from modules.plotting import plot_labels

## ------------- Get parameters --------------
with open('data_generation_params.yaml') as file:
    p  =  yaml.safe_load(file)

filename = p['raw_data_path'] + p['data_filename'] + '.npz'

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


## --------------- Normalization (we normalize the material constants, frequency, load radius and mesh)

r_field = np.linspace(r_min, r_max, n)
z_field = np.linspace(z_min, z_max, m)

wd = np.zeros((n,m, num_freqs), dtype=complex)

## -------------- Calling function ----------------
for k in tqdm(range(len(freqs)), colour='Green'):
    for i in range(len(r_field)):
        for j in range(len(z_field)):
            wd[i, j, k] = (loadmag/np.pi*r_max**2)*influence(
                            c11, c12, c13, c33, c44,
                            dens, damp,
                            r_field[i], z_field[j],
                            z_source, r_source, l_source,
                            freqs[k],
                            bvptype, loadtype, component
                        )

try:
    os.makedirs(filename, exist_ok=True)
except FileExistsError as e:
    print('Rewriting previous data file...')

np.savez(filename, freqs=freqs, r_field=r_field, z_field=z_field, wd=wd)

## ----------- Plot -------------
R, Z = r_field, z_field
f_index = 0
wd_transposed = wd.transpose(2,1,0)
wd_f = wd_transposed[f_index]
wd_plot = wd_f
plot_labels(R,Z,wd_plot, freqs[f_index], plot_type='abs')
