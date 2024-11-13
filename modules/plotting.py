import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def plot_training(epochs, history):
    loss_data, error_data = history['loss'], history['error']
    learning_rate = history['learning_rate']

    train_loss = loss_data['train']
    val_loss = loss_data['val']

    train_error_real = error_data['train']['real']
    train_error_imag = error_data['train']['imag']
    val_error_real = error_data['val']['real']
    val_error_imag = error_data['val']['imag']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    ax[0].plot(epochs, train_loss, label='train_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_xlabel('epoch')
    ax[0].set_yscale('log')
    ax[0].set_title(r'Loss for $u_{zz}$')
    ax[0].legend()

    ax_0_sec = ax[0].twinx()
    ax_0_sec.plot(epochs, learning_rate, "k--", label='lr', linewidth=0.5)
    ax_0_sec.set_ylabel(r"learning_rate")
    ax_0_sec.set_yscale(r"log")


    ax[1].plot(epochs, train_error_real, label='real_train')
    ax[1].plot(epochs, val_error_real, label='real_val')
    ax[1].set_xlabel('epoch')
    ax[1].set_yscale('log')
    ax[1].set_title(r'$L_2$ Error for $\Re(u_{zz})$')
    ax[1].legend()
    ax_1_sec = ax[1].twinx()
    ax_1_sec.plot(epochs, learning_rate, "k--", label='lr', linewidth=0.5)
    ax_1_sec.set_ylabel(r"learning_rate")
    ax_1_sec.set_yscale(r"log")

    ax[2].plot(epochs, train_error_imag, label='imag_train')
    ax[2].plot(epochs, val_error_imag, label='imag_val')
    ax[2].set_xlabel('epoch')
    ax[2].set_yscale('log')
    ax[2].set_title(r'$L_2$ Error for $\Im(u_{zz})$')
    ax[2].legend()
    ax_2_sec = ax[2].twinx()
    ax_2_sec.plot(epochs, learning_rate, "k--", label='lr', linewidth=0.5)
    ax_2_sec.set_ylabel(r"learning_rate")
    ax_2_sec.set_yscale(r"log")
    
    fig.tight_layout()

    return fig

def plot_field(r ,z, wd, freq, full, non_dim_plot=True):
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

def plot_axis(r ,z, wd, freq, non_dim_plot=True, axis=None):
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
            elif axis == 'r':
                var = r
                plane=z.min()
                mask = np.where(z == plane)[0].item()
                wd_plot = wd[ : ,mask]
            else:
                raise ValueError
        except ValueError:
            pass
        u_real = np.real(wd_plot)
        u_imag = np.imag(wd_plot)
        u_abs = np.abs(wd_plot)



        fig, ax = plt.subplots(nrows=1,
                        ncols=3,
                        figsize=(16, 4))
        
        if axis == 'z':

            ax[0].plot(u_real, var, '.-k')
            ax[0].invert_yaxis()
            ax[0].set_ylabel(z_label, fontsize=14)

            ax[1].plot(u_imag, var, '.-k')
            ax[1].invert_yaxis()
            ax[1].set_ylabel(z_label, fontsize=14)

            ax[2].plot(u_abs, var, '.-k')
            ax[2].invert_yaxis()
            ax[2].set_ylabel(z_label, fontsize=14)
        else:

            ax[0].plot(var, u_real, '.-k')
            ax[0].set_xlabel(r_label, fontsize=14)

            ax[1].plot(var, u_imag, '.-k')
            ax[1].set_xlabel(r_label, fontsize=14)

            ax[2].plot(var, u_abs, '.-k')
            ax[2].set_xlabel(r_label, fontsize=14)

        ax[0].set_title(l_real + plot_type_id)
        ax[1].set_title(l_imag + plot_type_id)
        ax[2].set_title(l_abs + plot_type_id)
    else:
        r_plane = z.min()
        mask_z = np.where(z == r_plane)[0].item()
        wd_plot_r = wd[ : , mask_z]

        z_plane = r.min()
        mask_r = np.where(r == z_plane)[0].item()
        wd_plot_z = wd[mask_r , : ]

        u_real_r = np.real(wd_plot_r)
        u_imag_r = np.imag(wd_plot_r)
        u_abs_r = np.abs(wd_plot_r)
        u_real_z = np.real(wd_plot_z)
        u_imag_z = np.imag(wd_plot_z)
        u_abs_z = np.abs(wd_plot_z)

        fig, ax = plt.subplots(nrows=2,
                            ncols=3,
                            figsize=(14, 10),
                            sharex='row')
        
        ax[0][0].plot(r, u_real_r, '.-k')
        ax[0][0].set_xlabel(r_label, fontsize=14)

        ax[0][1].plot(r, u_imag_r, '.-k')
        ax[0][1].set_xlabel(r_label, fontsize=14)

        ax[0][2].plot(r, u_abs_r, '.-k')
        ax[0][2].set_xlabel(r_label, fontsize=14)

        ax[1][0].plot(u_real_z, z, '.-k')
        ax[1][0].invert_yaxis()
        ax[1][0].set_ylabel(z_label, fontsize=14)

        ax[1][1].plot(u_imag_z, z, '.-k')
        ax[1][1].invert_yaxis()
        ax[1][1].set_ylabel(z_label, fontsize=14)

        ax[1][2].plot(u_abs_z, z, '.-k')
        ax[1][2].invert_yaxis()
        ax[1][2].set_ylabel(z_label, fontsize=14)

        ax[0][0].set_title(l_real[ : -1] + plot_type_r)
        ax[0][1].set_title(l_imag[ : -1] + plot_type_r)
        ax[0][2].set_title(l_abs[ : -1] + plot_type_r)
        ax[1][0].set_title(l_real[ : -1] + plot_type_z)
        ax[1][1].set_title(l_imag[ : -1] + plot_type_z)
        ax[1][2].set_title(l_abs[ : -1] + plot_type_z)
        
    fig.tight_layout()
    return fig

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

    plot_norm_abs = Normalize(vmin=plot_abs_min, vmax=plot_abs_max)
    plot_norm_real = Normalize(vmin=plot_real_min, vmax=plot_real_max)
    plot_norm_imag = Normalize(vmin=plot_imag_min, vmax=plot_imag_max)

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

def plot_labels_axis(r, z, wd_full, freqs, non_dim_plot=True):
    if not isinstance(freqs, np.ndarray) and not isinstance(freqs, list):
        freqs = freqs.detach().numpy()
    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'
    l_abs= r'$|u_{zz}|$'

    if non_dim_plot:
        r_label = r'$\frac{r}{a}$'
        z_label = r'$\frac{z}{a}$'
        plot_type_id = r'$a_{0} = $'
        plot_type_r = r'(r, z=0)$'
        plot_type_z = r'(r=0, z)$'
    else:
        r_label = r'$r$'
        z_label = r'$z$'
        plot_type_id = r'$\omega = $'
        plot_type_r = r'(r, z=0)$'
        plot_type_z = r'(r=0, z)$'

    r_plane = z.min()
    mask_z = np.where(z == r_plane)[0].item()
    wd_plot_r = wd_full[ : , : , mask_z]

    z_plane = r.min()
    mask_r = np.where(r == z_plane)[0].item()
    wd_plot_z = wd_full[ : , mask_r, : ]

    u_real_r = np.real(wd_plot_r)
    u_imag_r = np.imag(wd_plot_r)
    u_abs_r = np.abs(wd_plot_r)

    u_real_z = np.real(wd_plot_z)
    u_imag_z = np.imag(wd_plot_z)
    u_abs_z = np.abs(wd_plot_z)

    fig, ax = plt.subplots(nrows=2,
                    ncols=3,
                    figsize=(14, 10),
                    sharex='row')

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[0][0].plot(r, u_real_r[i , : ], label=plot_type_id + label)
    ax[0][0].set_xlabel(r_label, fontsize=14)
    ax[0][0].legend()

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[0][1].plot(r, u_imag_r[i , : ], label=plot_type_id + label)
    ax[0][1].set_xlabel(r_label, fontsize=14)
    ax[0][1].legend()

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[0][2].plot(r, u_abs_r[i , : ], label=plot_type_id + label)
    ax[0][2].set_xlabel(r_label, fontsize=14)
    ax[0][2].legend()

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[1][0].plot(z, u_real_z[i , : ], label=plot_type_id + label)
    ax[1][0].legend()
    ax[1][0].set_ylabel(z_label, fontsize=14)

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[1][1].plot(z, u_imag_z[i , : ], label=plot_type_id + label)
    ax[1][1].legend()
    ax[1][1].set_ylabel(z_label, fontsize=14)

    for i in range(len(freqs)):
        label = f" {freqs[i].item():.1f}"
        ax[1][2].plot(z, u_abs_z[i , : ], label=plot_type_id + label)
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