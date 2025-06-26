import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def plot_kelvin_solution(
    npz_path: str,
    sample_index: int = 0,
    stride: int = 1,
) -> None:
    """
    Load a .npz file produced by the Kelvin‐solution generator and plot
    the 3D displacement field (quiver) for one branch sample.

    Args:
        npz_path (str): Path to the .npz file containing:
                        F, mu, nu, x, y, z, g_u.
        sample_index (int): Index of the branch sample to visualize (default: 0).
        stride (int): Plot every 'stride' points along each axis to reduce clutter.
                      E.g., stride=2 will plot every other point in x,y,z.
    """
    # 1. Load data
    data = np.load(npz_path)
    x_field: np.ndarray = data["x"]         # shape (n_x,)
    y_field: np.ndarray = data["y"]         # shape (n_y,)
    z_field: np.ndarray = data["z"]         # shape (n_z,)
    g_u: np.ndarray = data["g_u"]           # shape (N, n_x, n_y, n_z, 3)

    N, n_x, n_y, n_z, _ = g_u.shape

    if sample_index < 0 or sample_index >= N:
        raise IndexError(f"sample_index {sample_index} out of bounds (0 ≤ i < {N}).")

    # 2. Create a meshgrid of all (x,y,z) points
    X, Y, Z = np.meshgrid(x_field, y_field, z_field, indexing="ij")  # each shape (n_x, n_y, n_z)

    # 3. Extract the displacement vectors for the chosen sample
    #    g_u has shape (N, n_x, n_y, n_z, 3). We take index sample_index:
    U_sample: np.ndarray = g_u[sample_index, :, :, :, :]  # shape (n_x, n_y, n_z, 3)

    # 4. Optionally subsample every `stride` points to avoid overcrowding
    Xs = X[::stride, ::stride, ::stride]
    Ys = Y[::stride, ::stride, ::stride]
    Zs = Z[::stride, ::stride, ::stride]
    Us = U_sample[::stride, ::stride, ::stride, :]

    # 5. Flatten for plotting
    #    Each of Xs, Ys, Zs, Us[...,0], Us[...,1], Us[...,2] will become 1D arrays
    X_flat = Xs.ravel()
    Y_flat = Ys.ravel()
    Z_flat = Zs.ravel()
    Ux_flat = Us[..., 0].ravel()
    Uy_flat = Us[..., 1].ravel()
    Uz_flat = Us[..., 2].ravel()

    # 6. Build a 3D quiver plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # quiver: (x, y, z, u, v, w), length=scale of arrows
    ax.quiver(
        X_flat,
        Y_flat,
        Z_flat,
        Ux_flat,
        Uy_flat,
        Uz_flat,
        length=0.1 * np.max([x_field.ptp(), y_field.ptp(), z_field.ptp()]),
        normalize=True,
        linewidth=0.5,
    )

    ax.set_title(f"Kelvin Displacement Field (sample {sample_index})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ─── USER CONFIGURATION ─────────────────────────────────────────────────────
    # Path to the .npz file you’ve saved in your generate() step:
    npz_filename = "/Users/ls/Workspace/SSI_DeepONet/data/raw/kelvin/kelvin_v1.npz"

    # Which branch‐sample index to plot (0 ≤ sample_index < N):
    sample_to_plot = 0

    # To reduce arrow density, you can set stride > 1 (e.g., stride=2 or 3),
    # otherwise keep stride=1 to plot every grid point.
    subsample_stride = 2
    # ─────────────────────────────────────────────────────────────────────────────

    # Verify file exists
    if not Path(npz_filename).is_file():
        raise FileNotFoundError(f"Cannot find file: {npz_filename}")

    plot_kelvin_solution(
        npz_path=npz_filename,
        sample_index=sample_to_plot,
        stride=subsample_stride,
    )
