import numpy as np
import matplotlib.pyplot as plt

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

    # Create a new figure and a 3D subplot
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

    # Set labels for the axes and the title of the plot
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)

    # Adjust the initial view angle for better visualization (optional)
    ax.view_init(elev=20, azim=-60)

    # Display the plot
    plt.show()


pod_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/pod.npz'
# processed_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/rajapakse_fixed_material/a37f8126/data.npz'
# data = np.load(processed_path)
# g_u = data['g_u']
pod_data = np.load(pod_path)
stacked_basis = pod_data['stacked_basis']
stacked_mean = pod_data['stacked_mean']
split_basis = pod_data['split_basis']
split_mean = pod_data['split_mean']
print(stacked_basis.shape, stacked_mean.shape,
      split_basis.shape, split_mean.shape)
# test = g_u.reshape(30, 20, 20, 2).transpose(0, 3, 2, 1)
# print(test.shape)
# fig = plot_basis_function(test[0][1])
# plt.show()


test_1 = split_basis.T
for i in range(test_1.shape[0]):
    print(test_1.shape)
    x_coords = np.arange(20)
    y_coords = np.arange(20)
    z_coords = np.arange(20)
    fig = plot_pod_mode_3d(x_coords, y_coords, z_coords, test_1[i].reshape(20, 20, 20).T)
    plt.show()
# fig = plot_basis_function(g_u_full[0].T.imag)
# plt.show()
plot_pod_mode_3d(
    x, y, z, single_basis_pod[:, 0], title="Single Basis POD Mode 0")
