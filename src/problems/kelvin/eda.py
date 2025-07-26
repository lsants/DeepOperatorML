import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from plot_kelvin_3d import plot_kelvin_solution

path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/data.npz'
raw_path = './data/raw/kelvin/kelvin_v4.npz'

data = np.load(path)
raw_data = dict(np.load(raw_path))

xb, xt, gu = data['xb'], data['xt'], data['g_u']
x, y, z = raw_data['x'], raw_data['y'], raw_data['z']

raw_gu = raw_data['g_u']

print(x.min(), x.max(), x.mean(), x.std())
print(y.min(), y.max(), y.mean(), y.std())
print(z.min(), z.max(), z.mean(), z.std())
print(gu[0].min(axis=0), gu[0].max(axis=0),
      gu[0].mean(axis=0), gu[0].std(axis=0))

for sample in gu:
    x_grid, y_grid, z_grid = np.mgrid[:len(x), :len(y), :len(z)]
    grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
    grid["disp"] = sample

    sample_to_plot = 0

    streams = grid.streamlines(
        vectors="disp",
        max_time=100,
        n_points=500,
        integration_direction='both'
    )

    plotter = pv.Plotter()
    plotter.add_mesh(streams.tube(radius=0.1), scalars="disp", lighting=False)
    plotter.add_mesh(grid.outline(), color='k')
    # plotter.add_mesh(grid.contour(isosurfaces=10, scalars="disp_x"))
    max_point = grid.points[np.argmax(np.linalg.norm(grid["disp"], axis=1))]
    plotter.add_mesh(pv.Sphere(radius=0.5, center=max_point), color='red')
    plotter.show()


# To reduce arrow density, you can set stride > 1 (e.g., stride=2 or 3),
# otherwise keep stride=1 to plot every grid point.

# subsample_stride = 3

# # ─────────────────────────────────────────────────────────────────────────────

# plot_kelvin_solution(
#     data=raw_data,
#     sample_index=sample_to_plot,
#     stride=subsample_stride,
# )
