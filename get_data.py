import yaml
import numpy as np
from tqdm.auto import tqdm
from data_generation.influence import influence
from modules.plotting import plot_labels

## ------------- Get parameters --------------
with open('data_generation_params.yaml') as file:
    p  =  yaml.safe_load(file)

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
freqs  =  np.random.rand(num_freqs) * p['freq_max']
r_max = p['r_max']
z_max = p['z_max']
z_fonte = p['z_fonte']
r_fonte = p['r_fonte']
l_fonte = p['l_fonte']
bvptype = p['bvptype']
loadtype = p['loadtype']
component = p['component']

r_campo = np.linspace(0,r_max, n)
z_campo = np.linspace(0, z_max, m)

wd = np.zeros((n,m, num_freqs), dtype=complex)

## -------------- Calling function ----------------
for k in tqdm(range(len(freqs)), colour='Green'):
    for i in range(len(r_campo)):
        for j in range(len(z_campo)):
            wd[i, j, k] = loadmag*influence(
                            c11, c12, c13, c33, c44,
                            dens, damp,
                            r_campo[i], z_campo[j],
                            z_fonte, r_fonte, l_fonte,
                            freqs[k],
                            bvptype, loadtype, component
                        )

## ----------- Plot -------------
R, Z = r_campo, z_campo
f_index = 0

wd_transposed = wd.transpose(2,1,0)
wd_f = wd_transposed[f_index]
wd_plot = wd_f

plot_labels(R,Z,wd_plot, freqs[f_index], plot_type='abs')
