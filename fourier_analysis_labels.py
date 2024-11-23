import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules import dir_functions
from modules import preprocessing as ppr
from modules.greenfunc_dataset import GreenFuncDataset
from modules.plotting import plot_fft_field

n = 2
f_index = 0

date = datetime.today().strftime('%Y%m%d')

with open('params_test.yaml') as file:
    p = yaml.safe_load(file)

data = np.load(p['DATAFILE'], allow_pickle=True)
dataset = GreenFuncDataset(data)
indices = dir_functions.load_indices(p['INDICES_FILE'])
norm_params = dir_functions.load_indices(p['NORM_PARAMS_FILE'])

print(f"Plotting data from: {p['DATAFILE']}")
print(f"Using indices from: {p['INDICES_FILE']}")

train_indices = indices['train']
train_dataset = dataset[train_indices]

xt = dataset.get_trunk()
r, z = ppr.trunk_to_meshgrid(xt)

xb = train_dataset['xb']
g_u_real = train_dataset['g_u_real']
g_u_imag = train_dataset['g_u_imag']

g_u = g_u_real + g_u_imag * 1j
g_u = ppr.reshape_from_model(g_u, z)

f_label = xb[f_index]
freq = f_label.item()
wd_plot = g_u[f_index]

if g_u.shape != (len(z), len(r)):
    g_u = g_u.T

u_fft = np.fft.fft2(g_u)

dr = r[1] - r[0]
dz = z[1] - z[0]

freq_r = np.fft.fftfreq(len(r), d=dr)
freq_z = np.fft.fftfreq(len(z), d=dz)

positive_freq_r_indices = freq_r >= 0
positive_freq_z_indices = freq_z >= 0

freq_r_positive = freq_r[positive_freq_r_indices]
freq_z_positive = freq_z[positive_freq_z_indices]

u_fft_positive = u_fft[np.ix_(positive_freq_z_indices, positive_freq_r_indices)]

print(u_fft.shape)

fft_real = u_fft_positive.real
fft_imag = u_fft_positive.imag

flattened_real = fft_real.flatten()
flattened_imag = fft_imag.flatten()

indices_real = np.argpartition(flattened_real, -n)[-n:]
sorted_indices_real = indices_real[np.argsort(flattened_real[indices_real])[ : : -1]]
indices_real_2d = np.array(np.unravel_index(sorted_indices_real, fft_real.shape)).T

indices_imag = np.argpartition(flattened_imag, -n)[-n:]
sorted_indices_imag = indices_imag[np.argsort(flattened_imag[indices_imag])[ : : -1]]
indices_imag_2d = np.array(np.unravel_index(sorted_indices_imag, fft_imag.shape)).T

top_freqs_real = []
for idx_pair in indices_real_2d:
    freq_z_idx, freq_r_idx = idx_pair
    freq_z_val = freq_z_positive[freq_z_idx]
    freq_r_val = freq_r_positive[freq_r_idx]
    fft_value = fft_real[freq_z_idx, freq_r_idx]
    top_freqs_real.append({
        'freq_r': freq_r_val,
        'freq_z': freq_z_val,
        'fft_value': fft_value
    })

top_freqs_imag = []
for idx_pair in indices_imag_2d:
    freq_z_idx, freq_r_idx = idx_pair
    freq_z_val = freq_z_positive[freq_z_idx]
    freq_r_val = freq_r_positive[freq_r_idx]
    fft_value = fft_imag[freq_z_idx, freq_r_idx]
    top_freqs_imag.append({
        'freq_r': freq_r_val,
        'freq_z': freq_z_val,
        'fft_value': fft_value
    })

max_freqs = {'real': top_freqs_real, 'imag': top_freqs_imag}

filename = f"top_{n}_frequencies_{p['MODELNAME']}.json"
with open(filename, 'w') as f:
    json.dump(max_freqs, f, indent=4)
print(f"Top {n} frequencies saved to {filename}")
