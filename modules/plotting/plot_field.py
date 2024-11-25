import numpy as np
import matplotlib.pyplot as plt

def plot_field(r ,z, wd, freq=None, full=True, non_dim_plot=True):
    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full,z)
        u_flip = np.flip(wd[1 : , : ], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)
        wd_full = wd_full.T
    else:
        R, Z = np.meshgrid(r,z)
        wd_full = wd.T

    u_real = np.real(wd_full)
    u_imag = np.imag(wd_full)
    u_abs = np.abs(wd_full)

    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'
    l_abs= r'|$u_{zz}$|'

    if non_dim_plot:
        title_real = l_real + r' at $a_0$' + f' = {freq:.2E}'
        title_imag = l_imag + r' at $a_0$' + f' = {freq:.2E}'
        title_abs = l_abs + r' at $a_0$' + f' = {freq:.2E}'
        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    
    else:
        title_real = l_real + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_imag = l_imag + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_abs = l_abs + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        x_label = r'$r$'
        y_label = r'$z$'


    fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(16, 4))
    
    contour_real = ax[0].contourf(R,Z, u_real, cmap="viridis")
    ax[0].invert_yaxis()
    ax[0].set_xlabel(x_label, fontsize=14)
    ax[0].set_ylabel(y_label, fontsize=14)
    ax[0].set_title(title_real)

    contour_imag = ax[1].contourf(R,Z, u_imag, cmap="viridis")
    ax[1].invert_yaxis()
    ax[1].set_xlabel(x_label, fontsize=14)
    ax[1].set_ylabel(y_label, fontsize=14)
    ax[1].set_title(title_imag)

    contour_abs = ax[2].contourf(R,Z,u_abs, cmap="viridis")
    ax[2].invert_yaxis()
    ax[2].set_xlabel(x_label, fontsize=14)
    ax[2].set_ylabel(y_label, fontsize=14)
    ax[2].set_title(title_abs)

    cbar_real = fig.colorbar(contour_real, label=l_real, ax=ax[0])
    cbar_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

    cbar_imag = fig.colorbar(contour_imag, label=l_imag, ax=ax[1])
    cbar_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    cbar_abs = fig.colorbar(contour_abs, label=l_abs, ax=ax[2])
    cbar_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig


