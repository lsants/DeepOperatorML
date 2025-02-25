import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def plot_fft_field(r, z, wd, freq, full=False, non_dim_plot=True):
    if full:
        r_full = np.concatenate((-np.flip(r[1:], axis=0), r))
        R, Z = np.meshgrid(r_full, z)
        u_flip = np.flip(wd[1:, :], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)
    else:
        r_full = r
        R, Z = np.meshgrid(r, z)
        wd_full = wd

    u = wd_full.T
    u_abs = np.abs(u)
    u_real = np.real(u)
    u_imag = np.imag(u)

    dr = r_full[1] - r_full[0]
    dz = z[1] - z[0]

    u_fft = np.fft.fft2(u_real + 1j * u_imag)

    freq_r = np.fft.fftfreq(len(r_full), d=dr)
    freq_z = np.fft.fftfreq(len(z), d=dz)

    if full:
        u_fft_shifted = np.fft.fftshift(u_fft)
        freq_r_shifted = np.fft.fftshift(freq_r)
        freq_z_shifted = np.fft.fftshift(freq_z)

        R_freq_grid, Z_freq_grid = np.meshgrid(freq_r_shifted, freq_z_shifted)

        u_fft_to_plot = u_fft_shifted
        R_freq_to_plot = R_freq_grid
        Z_freq_to_plot = Z_freq_grid
    else:
        positive_freq_r_indices = freq_r >= 0
        positive_freq_z_indices = freq_z >= 0

        freq_r_positive = freq_r[positive_freq_r_indices]
        freq_z_positive = freq_z[positive_freq_z_indices]

        u_fft_positive = u_fft[np.ix_(positive_freq_z_indices, positive_freq_r_indices)]

        R_freq_grid_positive, Z_freq_grid_positive = np.meshgrid(freq_r_positive, freq_z_positive)

        u_fft_to_plot = u_fft_positive
        R_freq_to_plot = R_freq_grid_positive
        Z_freq_to_plot = Z_freq_grid_positive

    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'
    l_abs = r'|$u_{zz}$|'

    l_real_fft = r'$\Re(\mathcal{F}\{u_{zz}\})$'
    l_imag_fft = r'$\Im(\mathcal{F}\{u_{zz}\})$'
    l_abs_fft = r'|$\mathcal{F}\{u_{zz}\}$|'

    if non_dim_plot:
        title_real = l_real + r' at $a_0$' + f' = {freq:.2E}'
        title_imag = l_imag + r' at $a_0$' + f' = {freq:.2E}'
        title_abs = l_abs + r' at $a_0$' + f' = {freq:.2E}'

        title_real_fft = l_real_fft + r' at $a_0$' + f' = {freq:.2E}'
        title_imag_fft = l_imag_fft + r' at $a_0$' + f' = {freq:.2E}'
        title_abs_fft = l_abs_fft + r' at $a_0$' + f' = {freq:.2E}'
        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    else:
        title_real = l_real + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_imag = l_imag + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_abs = l_abs + r' at $\omega$' + f' = {freq:.2E} Rad/s'

        title_real_fft = l_real_fft + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_imag_fft = l_imag_fft + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_abs_fft = l_abs_fft + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        x_label = r'$r$'
        y_label = r'$z$'

    x_label_fft = r'$\nu_{\frac{r}{a}}$'
    y_label_fft = r'$\nu_{\frac{z}{a}}$'

    fig, ax = plt.subplots(nrows=2,
                           ncols=3,
                           figsize=(16, 8),
                           projection='3d')

    contour_real = ax[0][0].plot_surface(R, Z, u_real, cmap="viridis")
    ax[0][0].invert_yaxis()
    ax[0][0].set_xlabel(x_label, fontsize=14)
    ax[0][0].set_ylabel(y_label, fontsize=14)
    ax[0][0].set_title(title_real)

    contour_imag = ax[0][1].plot_surface(R, Z, u_imag, cmap="viridis")
    ax[0][1].invert_yaxis()
    ax[0][1].set_xlabel(x_label, fontsize=14)
    ax[0][1].set_ylabel(y_label, fontsize=14)
    ax[0][1].set_title(title_imag)

    contour_abs = ax[0][2].plot_surface(R, Z, u_abs, cmap="viridis")
    ax[0][2].invert_yaxis()
    ax[0][2].set_xlabel(x_label, fontsize=14)
    ax[0][2].set_ylabel(y_label, fontsize=14)
    ax[0][2].set_title(title_abs)

    contour_real_fft = ax[1][0].plot_surface(R_freq_to_plot, Z_freq_to_plot, u_fft_to_plot.real, cmap="viridis")
    ax[1][0].invert_yaxis()
    ax[1][0].set_xlabel(x_label_fft, fontsize=14)
    ax[1][0].set_ylabel(y_label_fft, fontsize=14)
    ax[1][0].set_title(title_real_fft)

    contour_imag_fft = ax[1][1].plot_surface(R_freq_to_plot, Z_freq_to_plot, u_fft_to_plot.imag, cmap="viridis")
    ax[1][1].invert_yaxis()
    ax[1][1].set_xlabel(x_label_fft, fontsize=14)
    ax[1][1].set_ylabel(y_label_fft, fontsize=14)
    ax[1][1].set_title(title_imag_fft)

    contour_abs_fft = ax[1][2].plot_surface(R_freq_to_plot, Z_freq_to_plot, np.abs(u_fft_to_plot), cmap="viridis")
    ax[1][2].invert_yaxis()
    ax[1][2].set_xlabel(x_label_fft, fontsize=14)
    ax[1][2].set_ylabel(y_label_fft, fontsize=14)
    ax[1][2].set_title(title_abs_fft)

    cbar_real = fig.colorbar(contour_real, ax=ax[0][0])
    cbar_imag = fig.colorbar(contour_imag, ax=ax[0][1])
    cbar_abs = fig.colorbar(contour_abs, ax=ax[0][2])
    cbar_real_fft = fig.colorbar(contour_real_fft, ax=ax[1][0])
    cbar_imag_fft = fig.colorbar(contour_imag_fft, ax=ax[1][1])
    cbar_abs_fft = fig.colorbar(contour_abs_fft, ax=ax[1][2])

    fig.tight_layout()
    return fig
