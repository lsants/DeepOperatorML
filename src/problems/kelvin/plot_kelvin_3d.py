import numpy as np
import pyvista as pv

raw_path = './data/raw/kelvin/kelvin_v4.npz'
raw_data = np.load(raw_path)

all_vectors = raw_data['g_u']
n_samples = all_vectors.shape[0]

sample_index_to_plot = 1
vector_field_3d = all_vectors[sample_index_to_plot]

x_coords = raw_data['x']
y_coords = raw_data['y']
z_coords = raw_data['z']

X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

pl = pv.Plotter(window_size=(1500, 900))

full_grid = pv.StructuredGrid(X, Y, Z)
pl.add_mesh(full_grid.outline(), color='gray', line_width=1)

stride = 2

X_sub = X[::stride, ::stride, ::stride]
Y_sub = Y[::stride, ::stride, ::stride]
Z_sub = Z[::stride, ::stride, ::stride]
sub_grid = pv.StructuredGrid(X_sub, Y_sub, Z_sub)

sub_vectors_3d = vector_field_3d[::stride, ::stride, ::stride, :]


sub_vectors_flat = sub_vectors_3d.reshape(-1, 3)

sub_grid['vectors'] = sub_vectors_flat

sub_grid['magnitude'] = np.linalg.norm(sub_vectors_flat, axis=1)

arrows = sub_grid.glyph(
    orient='vectors',
    scale=False,
    factor=0.1,
    geom=pv.Arrow()
)

pl.add_mesh(
    arrows,
    cmap='viridis',
    show_scalar_bar=True,
    scalar_bar_args={'title': 'Vector Magnitude'}
)

pl.add_axes(
    xlabel='X',
    ylabel='Y',
    zlabel='Z'
)

# axes = pl.add_axes_at_origin(
#     x_color='red',   # X‐axis in red
#     y_color='green', # Y‐axis in green
#     z_color='blue',  # Z‐axis in blue
#     line_width=0.1
# )

# for caption_actor in (
#     axes.GetXAxisCaptionActor2D(),
#     axes.GetYAxisCaptionActor2D(),
#     axes.GetZAxisCaptionActor2D(),
# ):
#     txt_prop = caption_actor.GetCaptionTextProperty()
#     txt_prop.SetFontSize(3)        # shrink this number to taste
#     txt_prop.BoldOff()              # if you want them normal weight



axis_length = 1e-3

arrow_kwargs = dict(
    tip_length=0.05,    # fraction of total length
    tip_radius=0.02,
    shaft_radius=0.005,
    scale=1   # scale the whole arrow to `axis_length`
)


# X–axis (red)
arrow_x = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), **arrow_kwargs)
pl.add_mesh(arrow_x, color='red', name='X axis')

letter = 'x'
dir_arr = np.array((1,0,0)) / np.linalg.norm((1,0,0))
tip_pos = dir_arr * axis_length * (1 + arrow_kwargs['tip_length'] + 0.02)

# txt_x = pv.Text3D(letter,
#                   depth=axis_length*0.02,
#                   center=tip_pos,
#                   direction=
# #                 )
# pl.add_mesh(txt_x, color='black', name='X axis')


# Y–axis (green)
arrow_y = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), **arrow_kwargs)
pl.add_mesh(arrow_y, color='green', name='Y axis')

# Z–axis (blue)
arrow_z = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), **arrow_kwargs)
pl.add_mesh(arrow_z, color='blue', name='Z axis')

pl.camera_position = 'iso'
pl.show()