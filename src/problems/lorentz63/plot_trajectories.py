from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from matplotlib.figure import Figure

logger = logging.getLogger(__file__)

plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_lorenz_trajectories_1d(
    trajectories: np.ndarray,
    trajectory_window: np.ndarray,
    initial_conditions: np.ndarray,
    num_to_plot: int = 1,
    random_selection: bool = False
    ):
    """
    Plots 1D component-wise trajectories of the Lorenz system.
    
    Creates a 2D plot showing x, y, z coordinates vs. time in separate subplots.
    
    Args:
        trajectories (np.ndarray): A 3D array of shape (num_trajectories, num_time_points, 3)
            containing the (x, y, z) coordinates over time.
        trajectory_window (np.ndarray): A 1D array of time points corresponding to the
            second dimension of the trajectories array.
        initial_conditions (np.ndarray): A 2D array of shape (num_trajectories, 3)
            containing the (x0, y0, z0) for each trajectory.
        num_to_plot (int): The number of trajectories to display on the plots.
        random_selection (bool): If True, selects trajectories randomly. Otherwise, plots
            the first `num_to_plot` trajectories.
    """
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        logger.error("`trajectories` array must be of shape (n, t, 3).")
        return
    
    # logger.info(f"Generating 1D component plots for {num_to_plot} trajectories...")
    
    num_available = trajectories.shape[0]
    if random_selection:
        indices = np.random.choice(num_available, size=min(num_to_plot, num_available), replace=False)
    else:
        indices = np.arange(min(num_to_plot, num_available))
    
    # --- Figure: Coordinates vs. Time Plot (like MATLAB's plot) ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        # sharey=True,
        figsize=(12, 9), 
    )
    fig.suptitle(f"Trajectory Coordinates vs. Time")
    
    for i in indices:
        x, y, z = trajectories[i].T
        label = "$\\mathbf{r}_0$" + f"=({np.round(initial_conditions[i][0], 1)}, {np.round(initial_conditions[i][1], 1)}, {np.round(initial_conditions[i][2], 1)})" if len(indices) <= 10 else None
        ax1.plot(trajectory_window, x, lw=1.2, label=label)
        ax2.plot(trajectory_window, y, lw=1.2, label=label)
        ax3.plot(trajectory_window, z, lw=1.2, label=label)
    
    ax1.set_ylabel("$x$", fontsize=20)
    ax1.grid(True)
    if len(indices) <= 10:
        ax1.legend()
    
    ax2.set_ylabel("$y$", fontsize=20)
    ax2.grid(True)
    
    ax3.set_ylabel("$z$", fontsize=20)
    ax3.set_xlabel("$t$ [s]", fontsize=20)
    ax3.grid(True)
    
    plt.tight_layout()
    
    return fig

def plot_lorenz_1d_comparison(
    pred_trajectories: np.ndarray,
    true_trajectories: np.ndarray,
    trajectory_window: np.ndarray,
    initial_conditions: np.ndarray,
    num_to_plot: int = 1,
    random_selection: bool = False
):
    """
    Plots 1D component-wise trajectories of the Lorenz system.
    
    Creates a 2D plot showing x, y, z coordinates vs. time in separate subplots.
    
    Args:
        trajectories (np.ndarray): A 3D array of shape (num_trajectories, num_time_points, 3)
            containing the (x, y, z) coordinates over time.
        trajectory_window (np.ndarray): A 1D array of time points corresponding to the
            second dimension of the trajectories array.
        initial_conditions (np.ndarray): A 2D array of shape (num_trajectories, 3)
            containing the (x0, y0, z0) for each trajectory.
        num_to_plot (int): The number of trajectories to display on the plots.
        random_selection (bool): If True, selects trajectories randomly. Otherwise, plots
            the first `num_to_plot` trajectories.
    """
    if true_trajectories.ndim != 3 or true_trajectories.shape[2] != 3 or true_trajectories.ndim != 3 or pred_trajectories.shape[2] != 3:
        logger.error("`trajectories` array must be of shape (n, t, 3).")
        return
    
    # logger.info(f"Generating 1D component plots for {num_to_plot} trajectories...")
    
    # --- Figure: Coordinates vs. Time Plot (like MATLAB's plot) ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        # sharey=True,
        figsize=(12, 9), 
    )
    fig.suptitle(f"Trajectory Coordinates vs. Time")
    
    x_pred, y_pred, z_pred = pred_trajectories.T
    x_true, y_true, z_true = true_trajectories.T
    label_pred = "Predicted $\\mathbf{r}_0$" + f"=({np.round(initial_conditions[0], 1)}, {np.round(initial_conditions[1], 1)}, {np.round(initial_conditions[2], 1)})"
    label_true = "True $\\mathbf{r}_0$" + f"=({np.round(initial_conditions[0], 1)}, {np.round(initial_conditions[1], 1)}, {np.round(initial_conditions[2], 1)})"
    ax1.plot(trajectory_window, x_pred, '.-', lw=1.2, label=label_pred, color='magenta')
    ax1.plot(trajectory_window, x_true, lw=1.2, label=label_true, color='k')
    
    ax2.plot(trajectory_window, y_pred, '.-', lw=1.2, label=label_pred, color='purple')
    ax2.plot(trajectory_window, y_true, lw=1.2, label=label_true, color='k')
    
    ax3.plot(trajectory_window, z_pred, '.-', lw=1.2, label=label_pred, color='blue')
    ax3.plot(trajectory_window, z_true, lw=1.2, label=label_true, color='k')

    ax1.set_ylabel("$x$", fontsize=20)
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_ylabel("$y$", fontsize=20)
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_ylabel("$z$", fontsize=20)
    ax3.set_xlabel("$t$ [s]", fontsize=20)
    ax3.legend()
    ax3.grid(True)

    # plt.show()
    # quit()
    
    plt.tight_layout()
    
    return fig


def plot_lorenz_trajectories_3d(
    trajectories: np.ndarray,
    initial_conditions: np.ndarray,
    trajectory_window: np.ndarray,
    num_to_plot: int = 1,
    random_selection: bool = False
):
    """
    Plots 3D trajectories of the Lorenz system (the Lorenz attractor).
    
    Creates a 3D plot showing the trajectory paths in phase space.
    
    Args:
        trajectories (np.ndarray): A 3D array of shape (num_trajectories, num_time_points, 3)
            containing the (x, y, z) coordinates over time.
        initial_conditions (np.ndarray): A 2D array of shape (num_trajectories, 3)
            containing the (x0, y0, z0) for each trajectory.
        num_to_plot (int): The number of trajectories to display on the plot.
        random_selection (bool): If True, selects trajectories randomly. Otherwise, plots
            the first `num_to_plot` trajectories.
    """
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        logger.error("`trajectories` array must be of shape (n, t, 3).")
        return
    
    # logger.info(f"Generating 3D plot for {num_to_plot} trajectories...")
    
    num_available = trajectories.shape[0]
    if random_selection:
        indices = np.random.choice(num_available, size=min(num_to_plot, num_available), replace=False)
    else:
        indices = np.arange(min(num_to_plot, num_available))
    
    # --- Figure: 3D Trajectory Plot (like MATLAB's plot3) ---
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(indices) == 1:
        x, y, z = trajectories.T
        label = "$\\mathbf{r}_0$" + f"=({initial_conditions[0]:.1f}, {initial_conditions[1]:.1f}, {initial_conditions[2]:.1f})"
        ax.plot(x, y, z, lw=1.2, label=label)
        ax.set_title(f"Lorenz Attractor ($t$={trajectory_window[-1]} s)")
        ax.set_xlabel("$x$", fontsize=20)
        ax.set_ylabel("$y$", fontsize=20)
        ax.set_zlabel("$z$", fontsize=20)
        # ax.grid(True)
    else:
        for i in indices:
            x, y, z = trajectories[i].T
            label = "$\\mathbf{r}_0$" + f"=({initial_conditions[i][0]:.1f}, {initial_conditions[i][1]:.1f}, {initial_conditions[i][2]:.1f})"
            ax.plot(x, y, z, lw=1.2, label=label)
        
        ax.set_title(f"Lorenz Attractor ($t$={trajectory_window[-1]} s)")
        ax.set_xlabel("$x$", fontsize=20)
        ax.set_ylabel("$y$", fontsize=20)
        ax.set_zlabel("$z$", fontsize=20)
        ax.grid(True)
    
    if len(indices) <= 10:
        ax.legend()
    
    return fig


def plot_lorenz_trajectories(
    pred_trajectories: np.ndarray,
    true_trajectories: np.ndarray,
    trajectory_window: np.ndarray,
    initial_conditions: np.ndarray,
    num_to_plot: int = 1,
    random_selection: bool = False,
    plot_1d: bool = True,
    plot_3d: bool = True
) -> tuple[Figure | None, Figure | None]:
    """
    Plots Lorenz system trajectories from pre-processed data.
    
    This function can create both 1D component-wise plots and 3D trajectory plots.
    
    Args:
        trajectories (np.ndarray): A 3D array of shape (num_trajectories, num_time_points, 3)
            containing the (x, y, z) coordinates over time.
        trajectory_window (np.ndarray): A 1D array of time points corresponding to the
            second dimension of the trajectories array.
        initial_conditions (np.ndarray): A 2D array of shape (num_trajectories, 3)
            containing the (x0, y0, z0) for each trajectory.
        num_to_plot (int): The number of trajectories to display on the plots.
        random_selection (bool): If True, selects trajectories randomly. Otherwise, plots
            the first `num_to_plot` trajectories.
        plot_1d (bool): If True, creates 1D component-wise plots.
        plot_3d (bool): If True, creates 3D trajectory plot.
    """
    if plot_1d:
        if num_to_plot == 1:
            fig_1d = plot_lorenz_1d_comparison(
                pred_trajectories=pred_trajectories, 
                true_trajectories=true_trajectories, 
                trajectory_window=trajectory_window, 
                initial_conditions=initial_conditions, 
                num_to_plot=num_to_plot, 
            )
        else:
            fig_1d = plot_lorenz_trajectories_1d(
                trajectories=pred_trajectories, 
                trajectory_window=trajectory_window, 
                initial_conditions=initial_conditions, 
                num_to_plot=num_to_plot, 
            )
    
    if plot_3d:
        fig_3d = plot_lorenz_trajectories_3d(
            trajectories=pred_trajectories,
            initial_conditions=initial_conditions, 
            trajectory_window=trajectory_window,
            num_to_plot=num_to_plot
        )
        
    return fig_1d if fig_1d else None, fig_3d if fig_3d else None