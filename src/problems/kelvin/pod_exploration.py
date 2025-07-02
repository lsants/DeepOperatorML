import numpy as np
import matplotlib.pyplot as plt

path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin/f83e9497/data.npz'
raw_path = './data/raw/kelvin/kelvin_v4.npz'

data = np.load(path)
raw_data = np.load(raw_path)
var_share = 0.95

xb, xt, gu = data['xb'], data['xt'], data['g_u']
x, y, z = raw_data['x'], raw_data['y'], raw_data['z']

func_samples = gu.shape[0]
domain_samples = gu.shape[1]
num_channels = gu.shape[-1]


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
                         c=mode_values, cmap='viridis', s=20, alpha=0.7)

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


def single_basis_pod(data: np.ndarray) -> tuple[np.ndarray, ...]:
    domain_samples = data.shape[1]
    data_stacked = data.reshape(-1, domain_samples)

    mean = np.mean(data, axis=0, keepdims=True)
    mean_stacked = np.mean(data_stacked, axis=0, keepdims=True)
    centered = (data_stacked - mean_stacked).T

    U, S, _ = np.linalg.svd(centered, full_matrices=False)

    explained_variance_ratio = np.cumsum(
        S**2) / np.linalg.norm(S, ord=2)**2

    n_modes = (explained_variance_ratio < var_share).sum().item()

    single_basis_modes = U[:, : n_modes + 1]
    return single_basis_modes, mean


def multi_basis_pod(data: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(data, axis=0, keepdims=True)
    centered = (data - mean).transpose(1, 0, 2)

    centered_channels_first = centered.transpose(2, 0, 1)
    U, S, _ = np.linalg.svd(centered_channels_first, full_matrices=False)
    U = U.transpose(1, 2, 0)

    explained_variance_ratio = np.cumsum(
        S**2, axis=1).transpose(1, 0) / np.linalg.norm(S, axis=1, ord=2)**2

    modes_from_variance = (
        explained_variance_ratio <= var_share).sum().item()

    n_modes = modes_from_variance if modes_from_variance > 0  \
        else max(np.argmax(explained_variance_ratio, axis=0))

    multi_basis_modes = U[:, : n_modes + 1, :].transpose(0, 2, 1)
    multi_basis_modes = multi_basis_modes.reshape(
        multi_basis_modes.shape[0], -1)
    return multi_basis_modes, mean


single_basis_pod, _ = single_basis_pod(gu)
multi_basis_pod, _ = multi_basis_pod(gu)

print(single_basis_pod.shape)
print(multi_basis_pod.shape)

plot_pod_mode_3d(
    x, y, z, single_basis_pod[:, 0], title="Single Basis POD Mode 0")
