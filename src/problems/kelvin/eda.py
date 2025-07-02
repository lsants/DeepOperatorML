import numpy as np
import matplotlib.pyplot as plt
from plot_kelvin_3d import plot_kelvin_solution

path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/data.npz'
raw_path = './data/raw/kelvin/kelvin_v4.npz'

data = np.load(path)
raw_data = dict(np.load(raw_path))

xb, xt, gu = data['xb'], data['xt'], data['g_u']
x, y, z = raw_data['x'], raw_data['y'], raw_data['z']

print(x.min(), x.max(), x.mean(), x.std())
print(y.min(), y.max(), y.mean(), y.std())
print(z.min(), z.max(), z.mean(), z.std())
print(gu[0].min(axis=0), gu[0].max(axis=0),
      gu[0].mean(axis=0), gu[0].std(axis=0))

sample_to_plot = 0

# To reduce arrow density, you can set stride > 1 (e.g., stride=2 or 3),
# otherwise keep stride=1 to plot every grid point.
subsample_stride = 2

# ─────────────────────────────────────────────────────────────────────────────

plot_kelvin_solution(
    data=raw_data,
    sample_index=sample_to_plot,
    stride=subsample_stride,
)
