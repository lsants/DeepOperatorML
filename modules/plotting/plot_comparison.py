import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_field_comparison(r, z, wd, g_u, freq, full=True, non_dim_plot=True):
    R, Z = np.meshgrid(r, z)
    if full:
        r_full = np.concatenate((-np.flip(r[1 : ]), r))
        R, Z = np.meshgrid(r_full, z)

        wd_flip = np.flip(wd[1 : , : ], axis=0)
        wd = np.concatenate((wd_flip, wd), axis=0)

    wd_real = wd.real
    wd_imag = wd.imag
    wd_abs = abs(wd)

    if full:
        g_u_flip = np.flip(g_u[1 : , : ], axis=0)
        g_u = np.concatenate((g_u_flip, g_u), axis=0)

    g_u_real = g_u.real
    g_u_imag = g_u.imag
    g_u_abs = abs(g_u)

    print('\n')

    #  --------------- Setting axes labels and scales ----------------

    # For titles
    l_abs_pred = r'$|u_{z}|_{\mathrm{Pred}}$'
    l_real_pred = r'$\Re(u_{z})_{\mathrm{Pred}}$'
    l_imag_pred = r'$\Im(u_{z})_{\mathrm{Pred}}$'

    l_abs_label = r'$|u_{z}|_{\mathrm{Label}}$'
    l_real_label = r'$\Re(u_{z})_{\mathrm{Label}}$'
    l_imag_label = r'$\Im(u_{z})_{\mathrm{Label}}$'

    err_title = 'Absolute error for '

    # For axes
    if non_dim_plot:
        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
        plot_type_id = r' at $a_{0}$' + f'= {freq:.2E}'
    else:
        x_label = r'$r$'
        y_label = r'$z$'
        plot_type_id = r' at $\omega$' + f' = {freq:.2E} Rad/s'

    # For colorbar
    l_abs= r'|$u_{zz}$|'
    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'

    # For scales
    plot_abs_min = min(np.min(g_u_abs), np.min(wd_abs))
    plot_abs_max = max(np.max(g_u_abs), np.max(wd_abs))

    plot_real_min = min(np.min(g_u_real), np.min(wd_real))
    plot_real_max = max(np.max(g_u_real), np.max(wd_real))

    plot_imag_min = min(np.min(g_u_imag), np.min(wd_imag))
    plot_imag_max = max(np.max(g_u_imag), np.max(wd_imag))

    plot_norm_abs = colors.Normalize(vmin=plot_abs_min, vmax=plot_abs_max)
    plot_norm_real = colors.Normalize(vmin=plot_real_min, vmax=plot_real_max)
    plot_norm_imag = colors.Normalize(vmin=plot_imag_min, vmax=plot_imag_max)

    # Defining figure
    fig, ax = plt.subplots(nrows=3,
                        ncols=3,
                        figsize=(14, 10),
                        sharex='row',
                        sharey='row')

    contour_preds_real = ax[0][0].contourf(R, Z, g_u_real.T, cmap="viridis", norm=plot_norm_real)
    ax[0][0].invert_yaxis()
    ax[0][0].set_xlabel(x_label)
    ax[0][0].set_ylabel(y_label)
    ax[0][0].set_title(l_real_pred + plot_type_id)

    contour_labels_real = ax[0][1].contourf(R, Z, wd_real.T, cmap="viridis", norm=plot_norm_real)
    ax[0][1].set_title(l_real_label + plot_type_id)

    contour_errors_real = ax[0][2].contourf(R, Z, abs(wd_real.T - g_u_real.T), cmap="viridis")
    ax[0][2].set_title(err_title + l_real_label + plot_type_id)

    contour_preds_imag = ax[1][0].contourf(R, Z, g_u_imag.T, cmap="viridis", norm=plot_norm_imag)
    ax[1][0].invert_yaxis()
    ax[1][0].set_xlabel(x_label)
    ax[1][0].set_ylabel(y_label)
    ax[1][0].set_title(l_imag_pred + plot_type_id)

    contour_labels_imag = ax[1][1].contourf(R, Z, wd_imag.T, cmap="viridis", norm=plot_norm_imag)
    ax[1][1].set_title(l_imag_label + plot_type_id)

    contour_errors_imag = ax[1][2].contourf(R, Z, abs(wd_imag.T - g_u_imag.T), cmap="viridis")
    ax[1][2].set_title(err_title + l_imag_label + plot_type_id)

    contour_preds_abs = ax[2][0].contourf(R, Z, g_u_abs.T, cmap="viridis", norm=plot_norm_abs)
    ax[2][0].invert_yaxis()
    ax[2][0].set_xlabel(x_label)
    ax[2][0].set_ylabel(y_label)
    ax[2][0].set_title(l_abs_pred + plot_type_id)

    contour_labels_abs = ax[2][1].contourf(R, Z, wd_abs.T, cmap="viridis", norm=plot_norm_abs)
    ax[2][1].set_title(l_abs_label + plot_type_id)

    contour_errors_abs = ax[2][2].contourf(R, Z, abs(wd_abs.T - g_u_abs.T), cmap="viridis")
    ax[2][2].set_title(err_title + l_abs_label + plot_type_id)

    cbar_labels_real = fig.colorbar(contour_labels_real, label=l_real, ax=ax[0][1], norm=plot_norm_real)
    cbar_preds_real = fig.colorbar(contour_preds_real, label=l_real, ax=ax[0][0], norm=plot_norm_real)
    cbar_errors_real = fig.colorbar(contour_errors_real, label=l_real, ax=ax[0][2])
    cbar_labels_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
    cbar_preds_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
    cbar_errors_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

    cbar_labels_imag = fig.colorbar(contour_labels_imag, label=l_imag, ax=ax[1][1], norm=plot_norm_imag)
    cbar_preds_imag = fig.colorbar(contour_preds_imag, label=l_imag, ax=ax[1][0], norm=plot_norm_imag)
    cbar_errors_imag = fig.colorbar(contour_errors_imag, label=l_imag, ax=ax[1][2])
    cbar_labels_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)
    cbar_preds_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)
    cbar_errors_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    cbar_labels_abs = fig.colorbar(contour_labels_abs, label=l_abs, ax=ax[2][1], norm=plot_norm_abs)
    cbar_preds_abs = fig.colorbar(contour_preds_abs, label=l_abs, ax=ax[2][0], norm=plot_norm_abs)
    cbar_errors_abs = fig.colorbar(contour_errors_abs, label=l_abs, ax=ax[2][2])
    cbar_labels_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)
    cbar_preds_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)
    cbar_errors_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig

def plot_axis_comparison(r, z, wd, g_u, freq, axis=None, non_dim_plot=True, rotated_z=False):
    wd_plot = wd

    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'
    l_abs= r'$|u_{zz}|$'

    if non_dim_plot:
        r_label = r'$\frac{r}{a}$'
        z_label = r'$\frac{z}{a}$'
        plot_type_id = r' at $a_{0}$' + f'= {freq:.2E}'
        plot_type_r = r'(r, z=0)$ at $a_{0}$' + f'= {freq:.2E}'
        plot_type_z = r'(r=0, z)$ at $a_{0}$' + f'= {freq:.2E}'
    else:
        r_label = r'$r$'
        z_label = r'$z$'
        plot_type_id = r' at $\omega$' + f' = {freq:.2E} Rad/s'
        plot_type_r = r'(r, z=0)$  at $\omega$' + f' = {freq:.2E} Rad/s'
        plot_type_z = r'(r=0, z)$ at $\omega$' + f' = {freq:.2E} Rad/s'

    if axis is not None:
        try:
            if axis == 'z':
                var = z
                plane=r.min()
                mask = np.where(r == plane)[0].item()
                wd_plot = wd[mask]
                g_u_plot = g_u[mask]
            elif axis == 'r':
                var = r
                plane=z.min()
                mask = np.where(z == plane)[0].item()
                wd_plot = wd[ : , mask]
                g_u_plot = g_u[ : , mask]
            else:
                raise ValueError
        except ValueError:
            pass
        u_real_label = np.real(wd_plot)
        u_imag_label = np.imag(wd_plot)
        u_abs_label = np.abs(wd_plot)
        u_real_pred = np.real(g_u_plot)
        u_imag_pred = np.imag(g_u_plot)
        u_abs_pred = np.abs(g_u_plot)

        fig, ax = plt.subplots(nrows=1,
                        ncols=3,
                        figsize=(16, 4))
        
        if axis == 'z':
            if rotated_z:
                ax[0].plot(u_real_label, var, '.-k', label='u_label')
                ax[0].plot(u_real_pred, var, 'xr', label='u_pred')
                ax[0].invert_yaxis()
            else:
                ax[0].plot(var, u_real_label, '.-k', label='u_label')
                ax[0].plot(var, u_real_pred, 'xr', label='u_pred')
            ax[0].legend()
            ax[0].set_ylabel(z_label, fontsize=14)

            if rotated_z:
                ax[1].plot(u_real_label, var, '.-k', label='u_label')
                ax[1].plot(u_real_pred, var, 'xr', label='u_pred')
                ax[1].invert_yaxis()
            else:
                ax[1].plot(var, u_real_label, '.-k', label='u_label')
                ax[1].plot(var, u_real_pred, 'xr', label='u_pred')
            ax[1].legend()
            ax[1].set_ylabel(z_label, fontsize=14)

            if rotated_z:
                ax[2].plot(u_real_label, var, '.-k', label='u_label')
                ax[2].plot(u_real_pred, var, 'xr', label='u_pred')
                ax[2].invert_yaxis()
            else:
                ax[2].plot(var, u_real_label, '.-k', label='u_label')
                ax[2].plot(var, u_real_pred, 'xr', label='u_pred')
            ax[2].legend()
            ax[2].set_ylabel(z_label, fontsize=14)
        else:

            ax[0].plot(var, u_real_label, '.-k', label='u_label')
            ax[0].plot(var, u_real_pred, 'xr', label='u_pred')
            ax[0].set_xlabel(r_label, fontsize=14)
            ax[0].legend()

            ax[1].plot(var, u_imag_label, '.-k', label='u_label')
            ax[1].plot(var, u_imag_pred, 'xr', label='u_pred')
            ax[1].set_xlabel(r_label, fontsize=14)
            ax[1].legend()

            ax[2].plot(var, u_abs_label, '.-k', label='u_label')
            ax[2].plot(var, u_abs_pred, 'xr', label='u_pred')
            ax[2].set_xlabel(r_label, fontsize=14)
            ax[2].legend()

        ax[0].set_title(l_real + plot_type_id)
        ax[1].set_title(l_imag + plot_type_id)
        ax[2].set_title(l_abs + plot_type_id)
    else:
        r_plane = z.min()
        mask_z = np.where(z == r_plane)[0].item()
        wd_plot_r = wd[ : , mask_z]
        g_u_plot_r = g_u[ : , mask_z]

        z_plane = r.min()
        mask_r = np.where(r == z_plane)[0].item()
        wd_plot_z = wd[mask_r]
        g_u_plot_z = g_u[mask_r]

        u_real_r_label = np.real(wd_plot_r)
        u_real_r_pred = np.real(g_u_plot_r)

        u_imag_r_label = np.imag(wd_plot_r)
        u_imag_r_pred = np.imag(g_u_plot_r)

        u_abs_r_label = np.abs(wd_plot_r)
        u_abs_r_pred = np.abs(g_u_plot_r)

        u_real_z_label = np.real(wd_plot_z)
        u_real_z_pred = np.real(g_u_plot_z)

        u_imag_z_label = np.imag(wd_plot_z)
        u_imag_z_pred = np.imag(g_u_plot_z)

        u_abs_z_label = np.abs(wd_plot_z)
        u_abs_z_pred = np.abs(g_u_plot_z)

        fig, ax = plt.subplots(nrows=2,
                            ncols=3,
                            figsize=(14, 10),
                            sharex='row')
        
        ax[0][0].plot(r, u_real_r_label, '.-k', label='u_label')
        ax[0][0].plot(r, u_real_r_pred, 'xr', label='u_pred')
        ax[0][0].set_xlabel(r_label, fontsize=14)
        ax[0][0].legend()

        ax[0][1].plot(r, u_imag_r_label, '.-k', label='u_label')
        ax[0][1].plot(r, u_imag_r_pred, 'xr', label='u_pred')
        ax[0][1].set_xlabel(r_label, fontsize=14)
        ax[0][1].legend()

        ax[0][2].plot(r, u_abs_r_label, '.-k', label='u_label')
        ax[0][2].plot(r, u_abs_r_pred, 'xr', label='u_pred')
        ax[0][2].set_xlabel(r_label, fontsize=14)
        ax[0][2].legend()

        if rotated_z:
            ax[1][0].plot(u_real_z_label, z, '.-k', label='u_label')
            ax[1][0].plot(u_real_z_pred, z, 'xr', label='u_pred')
            ax[1][0].invert_yaxis()
        else:
            ax[1][0].plot(z, u_real_z_label, '.-k', label='u_label')
            ax[1][0].plot(z, u_real_z_pred, 'xr', label='u_pred')
        ax[1][0].legend()
        ax[1][0].set_ylabel(z_label, fontsize=14)

        if rotated_z:
            ax[1][1].plot(u_imag_z_label, z, '.-k', label='u_label')
            ax[1][1].plot(u_imag_z_pred, z, 'xr', label='u_pred')
            ax[1][1].invert_yaxis()
        else:
            ax[1][1].plot(z, u_imag_z_label, '.-k', label='u_label')
            ax[1][1].plot(z, u_imag_z_pred, 'xr', label='u_pred')
        ax[1][1].legend()
        ax[1][1].set_ylabel(z_label, fontsize=14)

        if rotated_z:
            ax[1][2].plot(u_abs_z_label, z, '.-k', label='u_label')
            ax[1][2].plot(u_abs_z_pred, z, 'xr', label='u_pred')
            ax[1][2].invert_yaxis()
        else:
            ax[1][2].plot(z, u_abs_z_label, '.-k', label='u_label')
            ax[1][2].plot(z, u_abs_z_pred, 'xr', label='u_pred')
        ax[1][2].legend()
        ax[1][2].set_ylabel(z_label, fontsize=14)

        ax[0][0].set_title(l_real[ : -1] + plot_type_r)
        ax[0][1].set_title(l_imag[ : -1] + plot_type_r)
        ax[0][2].set_title(l_abs[ : -1] + plot_type_r)
        ax[1][0].set_title(l_real[ : -1] + plot_type_z)
        ax[1][1].set_title(l_imag[ : -1] + plot_type_z)
        ax[1][2].set_title(l_abs[ : -1] + plot_type_z)
        
    fig.tight_layout()
    return fig