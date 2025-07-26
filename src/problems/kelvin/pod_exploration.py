import numpy as np
import os
import plotly.express as px
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import http.server
import socketserver
import webbrowser

def plot_pod_mode_3d(x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray,
                     mode_values: np.ndarray, title: str = "POD Mode 3D Plot"):
    """
    Plots a 3D POD mode as a scatter plot with color representing the mode value.

    Args:
        x_coords (np.ndarray): 1D array of x-coordinates (unique values along x-axis).
        y_coords (np.ndarray): 1D array of y-coordinates (unique values along y-axis).
        z_coords (np.ndarray): 1D array of z-coordinates (unique values along z-axis).
        mode_values (np.ndarray): 1D array of mode values, flattened to match the
                                  total number of points in the 3D grid
                                  (len(x_coords) * len(y_coords) * len(z_coords)).
        title (str, optional): Title of the plot. Defaults to "POD Mode 3D Plot".
    """
    # Create a 3D meshgrid from the 1D coordinate arrays
    # 'ij' indexing creates (len(x), len(y), len(z)) shaped arrays
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Flatten the meshgrid coordinates to match the flattened mode_values array
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Ensure mode_values has the correct shape to match the flattened coordinates
    if mode_values.shape[0] != X_flat.shape[0]:
        print(
            f"Warning: Mismatch in number of points. mode_values shape: {mode_values.shape[0]}, expected: {X_flat.shape[0]}")
        # Attempt to reshape mode_values if it's compatible, otherwise raise an error
        try:
            mode_values = mode_values.reshape(X_flat.shape[0])
        except ValueError:
            raise ValueError(
                f"mode_values shape {mode_values.shape} does not match "
                f"expected flattened grid shape {X_flat.shape[0]}. "
                "Please ensure mode_values is flattened correctly (e.g., mode_values = mode_values.flatten())."
            )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot
    # 'c' argument specifies the color of each point based on 'mode_values'
    # 'cmap' sets the colormap (e.g., 'viridis', 'jet', 'coolwarm')
    # 's' sets the size of the markers, 'alpha' sets their transparency
    scatter = ax.scatter(X_flat, Y_flat, Z_flat,
                         c=mode_values, cmap='viridis', alpha=0.7)

    # Add a color bar to the plot to indicate the mapping of colors to mode values
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Mode Value")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)

    # Adjust the initial view angle for better visualization (optional)
    ax.view_init(elev=20, azim=-60)

    # Display the plot
    plt.show()


pod_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/pod.npz'
processed_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/data.npz'
raw_path = './data/raw/kelvin/kelvin_v4.npz'

pod_data = np.load(pod_path)
stacked_basis = pod_data['stacked_basis']
stacked_mean = pod_data['stacked_mean']
split_basis = pod_data['split_basis']
split_mean = pod_data['split_mean']
print(stacked_basis.shape, stacked_mean.shape,
      split_basis.shape, split_mean.shape)

raw_data = np.load(raw_path)
data = np.load(processed_path)
g_u_original = data['g_u']
g_u = data['g_u'].reshape(500, -1)
num_time_steps = g_u_original.shape[0]
num_channels = g_u_original.shape[-1]
spatial_dims_product = np.prod(g_u_original.shape[1:-1])

xb, xt, gu = data['xb'], data['xt'], data['g_u']


# vector_field_to_plot = raw_data['g_u']

# x_coords = raw_data['x']
# y_coords = raw_data['y']
# z_coords = raw_data['z']

# n_x, n_y, n_z = len(x_coords), len(y_coords), len(z_coords)

# X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
# vector_field_to_plot = raw_data['g_u']

# pl = pv.Plotter(window_size=(1000, 800))

# grid = pv.StructuredGrid(X, Y, Z)
# grid["vectors"] = vector_field_to_plot

# pl.add_mesh(grid.outline(), color='gray', line_width=1)

# stride = 5

# sub_grid = grid.extract_subset(
#     (slice(0, n_x, stride), slice(0, n_y, stride), slice(0, n_z, stride)),
#     uniform_spacing=True
# )

# sub_vectors = sub_grid["vectors"]

# vector_magnitudes = np.linalg.norm(sub_vectors, axis=1)

# pl.add_arrows(
#     sub_grid.points,
#     sub_vectors,
#     mag=0.05,
#     scalars=vector_magnitudes,
#     cmap='viridis',
#     show_scalar_bar=True,
#     scalar_bar_args={'title': 'Vector Magnitude'}
# )

# pl.add_axes(
#     interactive=False,
#     xlabel='X-coordinate',
#     ylabel='Y-coordinate',
#     zlabel='Z-coordinate',
#     box=True,
# )

# axis_length = 0.2
# pl.add_axes_at_origin(
#     labels=['X', 'Y', 'Z'],
#     line_width=4,
#     tip_length=0.15,
#     shaft_length=axis_length,
#     label_size=(0.05, 0.05),
#     cone_radius=0.05
# )

# pl.camera_position = 'iso'

# pl.show()

pca = PCA(n_components=3)
# Initialize lists to store components and their corresponding channel labels
all_components = []
channel_labels = []
channels = ['x', 'y', 'z']




for i in range(num_channels):
    current_channel_data = g_u_original[:, ..., i]

    current_channel_data_reshaped = current_channel_data.reshape(num_time_steps, -1)

    components_channel_i = pca.fit_transform(current_channel_data_reshaped)

    # Store the components
    all_components.append(components_channel_i)

    # Create labels for this channel's components
    # Each label indicates which channel the PCA point belongs to
    channel_labels.extend([f'Channel {channels[i]}'] * num_time_steps)

# Concatenate all components into a single NumPy array
# The shape will be (num_time_steps * num_channels, 3)
final_components = np.vstack(all_components)



fig = px.scatter_3d(final_components,
                    x=0, y=1, z=2, # Specify columns for x, y, z
                    color=channel_labels,
                    labels={'0': 'x', '1': 'y', '2': 'z'}, # Custom labels
                    title="PCA Components 3D Plot")

# --- Save the Plotly figure to an HTML file ---
html_file_path = "pca_3d_plot.html"
fig.write_html(html_file_path, auto_open=False) # auto_open=False to open manually via server

print(f"Plotly figure saved to: {os.path.abspath(html_file_path)}")

PORT = 8000 # You can use any available port
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}/{html_file_path}")
    # Open the browser to the served file
    webbrowser.open_new_tab(f"http://localhost:{PORT}/{html_file_path}")

    # To keep the server running until you manually stop it (Ctrl+C)
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

# test = g_u.reshape(30, 20, 20, 2).transpose(0, 3, 2, 1)
# print(test.shape)
# fig = plot_basis_function(test[0][1])
# plt.show()


test_1 = split_basis.T.reshape(3, 20, 20, 20)
test_2 = (test_1.swapaxes(0, 3) - test_1.mean(axis=(1, 2, 3))) / test_1.std(axis=(1, 2, 3))
# for i in range(test_1.shape[0]):
#     print(test_2.shape)
#     x_coords = np.linspace(0, 1, 20)
#     y_coords = np.linspace(0, 1, 20)
#     z_coords = -np.linspace(0, 1, 20)
#     fig = plot_pod_mode_3d(x_coords, y_coords, z_coords, test_2[i])
#     plt.show()
# fig = plot_basis_function(g_u_full[0].T.imag)
# plt.show()
# print(stacked_basis)
# quit()
# x_coords = np.linspace(0, 1, 20)
# y_coords = np.linspace(0, 1, 20)
# z_coords = -np.linspace(0, 1, 20)
# U, V = np.meshgrid(test_1[0, :, 0, 0], test_1[0, 0, 0, :])
# fig, ax = plt.subplots()
# q = ax.quiver(x_coords, z_coords, U, V)
# plt.show()
# plot_pod_mode_3d(
#     x_coords, y_coords, z_coords, split_basis[:, 0], title="Single Basis POD Mode 0")
