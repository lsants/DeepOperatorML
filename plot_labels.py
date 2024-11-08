import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules import dir_functions
from modules import preprocessing as ppr
from modules.greenfunc_dataset import GreenFuncDataset
from modules.plotting import plot_axis, plot_field

f_index = 0

date = datetime.today().strftime('%Y%m%d')

with open('params_test.yaml') as file:
    p = yaml.safe_load(file)

# ----------------Get Labels -----------------
data = np.load(p['DATAFILE'], allow_pickle=True)
dataset = GreenFuncDataset(data)
indices = dir_functions.load_indices(p['INDICES_FILE'])
norm_params = dir_functions.load_indices(p['NORM_PARAMS_FILE'])

print(f"Plotting data from: {p['DATAFILE']}")
print(f"Using indices from: {p['INDICES_FILE']}")

test_indices = indices['test']
test_dataset = dataset[test_indices]

xt = dataset.get_trunk()
r, z = ppr.trunk_to_meshgrid(xt)

xb = test_dataset['xb']
g_u_real = test_dataset['g_u_real']
g_u_imag = test_dataset['g_u_imag']

g_u = g_u_real + g_u_imag * 1j
g_u = ppr.reshape_from_model(g_u, z)

print(g_u.shape)

f_label = xb[f_index]

freq = f_label.item()

wd_plot = g_u[f_index]

fig = plot_field(r, z, wd_plot, freq, full=False)
plt.show()

fig = plot_axis(r, z, wd_plot, freq)
plt.show()