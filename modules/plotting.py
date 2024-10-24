import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_label_contours(r ,z, wd, freq, full=True, non_dim_plot=False):
    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full,z)
        u_flip = np.flip(wd[1:, :], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)
    else:
        R, Z = np.meshgrid(r,z)
        wd_full = wd

    u_abs = np.abs(wd_full.T)
    u_real = np.real(wd_full.T)
    u_imag = np.imag(wd_full.T)

    l_abs= r'|$u_{zz}$|'
    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'

    if non_dim_plot:
        title_abs = l_abs + r' at $a_0$' + f' = {freq:.2E}'
        title_real = l_real + r' at $a_0$' + f' = {freq:.2E}'
        title_imag = l_imag + r' at $a_0$' + f' = {freq:.2E}'
        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    
    else:
        title_abs = l_abs + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_real = l_real + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        title_imag = l_imag + r' at $\omega$' + f' = {freq:.2E} Rad/s'
        x_label = r'$r$'
        y_label = r'$z$'


    fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(16, 4))
    
    contour_abs = ax[0].contourf(R,Z,u_abs, cmap="viridis")
    ax[0].invert_yaxis()
    ax[0].set_xlabel(x_label, fontsize=14)
    ax[0].set_ylabel(y_label, fontsize=14)
    ax[0].set_title(title_abs)

    contour_real = ax[1].contourf(R,Z, u_real, cmap="viridis")
    ax[1].invert_yaxis()
    ax[1].set_xlabel(x_label, fontsize=14)
    ax[1].set_ylabel(y_label, fontsize=14)
    ax[1].set_title(title_real)

    contour_imag = ax[2].contourf(R,Z, u_imag, cmap="viridis")
    ax[2].invert_yaxis()
    ax[2].set_xlabel(x_label, fontsize=14)
    ax[2].set_ylabel(y_label, fontsize=14)
    ax[2].set_title(title_imag)

    cbar_abs = fig.colorbar(contour_abs, label=l_abs, ax=ax[0])
    cbar_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

    cbar_real = fig.colorbar(contour_real, label=l_real, ax=ax[1])
    cbar_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

    cbar_imag = fig.colorbar(contour_imag, label=l_imag, ax=ax[2])
    cbar_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig

def plot_label_axis(r ,z, wd, freq, axis, non_dim_plot):
    wd_plot = wd
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
            wd_plot = wd[:,mask]
        else:
            raise ValueError
    except ValueError:
        pass

    u_abs = np.abs(wd_plot)
    u_real = np.real(wd_plot)
    u_imag = np.imag(wd_plot)

    l_abs= r'|$u_{zz}$|'
    l_real = r'$\Re(u_{zz})$'
    l_imag = r'$\Im(u_{zz})$'

    if non_dim_plot:
        r_label = r'$\frac{r}{a}$'
        z_label = r'$\frac{z}{a}$'
        plot_type_id = r' at $a_{0}$' + f'= {freq:.2E}'
    else:
        r_label = r'$r$'
        z_label = r'$z$'
        plot_type_id = r' at $\omega$' + f' = {freq:.2E} Rad/s'


    fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(16, 4))
    
    if axis == 'z':
        ax[0].plot(u_abs, var, '.-k')
        ax[0].invert_yaxis()
        ax[0].set_ylabel(z_label, fontsize=14)

        ax[1].plot(u_real, var, '.-k')
        ax[1].invert_yaxis()
        ax[1].set_ylabel(z_label, fontsize=14)

        ax[2].plot(u_imag, var, '.-k')
        ax[2].invert_yaxis()
        ax[2].set_ylabel(z_label, fontsize=14)
    else:
        ax[0].plot(var, u_abs, '.-k')
        ax[0].set_xlabel(r_label, fontsize=14)

        ax[1].plot(var, u_real, '.-k')
        ax[1].set_xlabel(r_label, fontsize=14)

        ax[2].plot(var, u_imag, '.-k')
        ax[2].set_xlabel(r_label, fontsize=14)

    ax[0].set_title(l_abs + plot_type_id)
    ax[1].set_title(l_real + plot_type_id)
    ax[2].set_title(l_imag + plot_type_id)

    return fig

def plot_vals(u_pred, xt, g_u_pred):
    pass

def plot_training(epochs, train_loss, train_error_real, train_error_imag, test_error_real, test_error_imag):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,9))

    ax[0][0].plot(epochs, [i for i in train_loss], label='train_z')
    ax[0][0].set_xlabel('epoch')
    ax[0][0].set_ylabel('MSE')
    ax[0][0].set_yscale('log')
    ax[0][0].set_title(r'$u_{zz}$ Loss')
    ax[0][0].legend()

    ax[0][1].plot(epochs, [np.sqrt(i**2 + j**2) for i,j in zip(train_error_real, train_error_imag)], label='abs_train')
    ax[0][1].plot(epochs, [np.sqrt(i**2 + j**2) for i,j in zip(test_error_real, test_error_imag)], label='abs_test')
    ax[0][1].set_xlabel('epoch')
    ax[0][1].set_ylabel(r'$L_2$ norm')
    ax[0][1].set_yscale('log')
    ax[0][1].set_title(r'Error for $|u_{zz}|$')
    ax[0][1].legend()

    ax[1][0].plot(epochs, [i for i in train_error_real], label='real_train')
    ax[1][0].plot(epochs, [i for i in test_error_real], label='real_test')
    ax[1][0].set_xlabel('epoch')
    ax[1][0].set_ylabel(r'$L_2$ norm')
    ax[1][0].set_yscale('log')
    ax[1][0].set_title(r'Error for $Re(u_{zz})$')
    ax[1][0].legend()

    ax[1][1].plot(epochs, [i for i in train_error_imag], label='imag_train')
    ax[1][1].plot(epochs, [i for i in test_error_imag], label='imag_test')
    ax[1][1].set_xlabel('epoch')
    ax[1][1].set_ylabel(r'$L_2$ norm')
    ax[1][1].set_yscale('log')
    ax[1][1].set_title(r'Error for $Im(u_{zz})$')
    ax[1][1].legend()

    plt.grid
    plt.show()

def plot_comparison(r, z, wd, g_u, freq, full=True, non_dim_plot=False):
    # Formatting labels
    R, Z = np.meshgrid(r, z)
    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full, z)

        wd_flip = np.flip(wd[1:, :], axis=0)
        wd = np.concatenate((wd_flip, wd), axis=0)

    wd_abs = abs(wd)
    wd_real = wd.real
    wd_imag = wd.imag

    # Formatting predictions
    if full:
        g_u_flip = np.flip(g_u[1:, :], axis=0)
        g_u = np.concatenate((g_u_flip, g_u), axis=0)

    g_u_abs = abs(g_u)
    g_u_real = g_u.real
    g_u_imag = g_u.imag

    print('\n')

    #  --------------- Setting axes labels and scales ----------------

    # For titles
    l_abs_pred = r'$|u_{z}|_{\mathrm{Pred}}$'
    l_real_pred = r'$\Re(u_{z})_{\mathrm{Pred}}$'
    l_imag_pred = r'$\Im(u_{z})_{\mathrm{Pred}}$'

    l_abs_label = r'$|u_{z}|_{\mathrm{Label}}$'
    l_real_label = r'$\Re(u_{z})_{\mathrm{Label}}$'
    l_imag_label = r'$\Im(u_{z})_{\mathrm{Label}}$'

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

    # Defining figure and plotting 
    fig, ax = plt.subplots(nrows=3,
                        ncols=2,
                        figsize=(14, 10),
                        sharex='row',
                        sharey='row')

    contour_preds_abs = ax[0][0].contourf(R, Z, g_u_abs.T, cmap="viridis", norm=plot_norm_abs)
    ax[0][0].invert_yaxis()
    ax[0][0].set_xlabel(x_label)
    ax[0][0].set_ylabel(y_label)
    ax[0][0].set_title(l_abs_pred + plot_type_id)

    contour_labels_abs = ax[0][1].contourf(R, Z, wd_abs.T, cmap="viridis", norm=plot_norm_abs)
    ax[0][1].set_title(l_abs_label + plot_type_id)

    contour_preds_real = ax[1][0].contourf(R, Z, g_u_real.T, cmap="viridis", norm=plot_norm_real)
    ax[1][0].invert_yaxis()
    ax[1][0].set_xlabel(x_label)
    ax[1][0].set_ylabel(y_label)
    ax[1][0].set_title(l_real_pred + plot_type_id)

    contour_labels_real = ax[1][1].contourf(R, Z, wd_real.T, cmap="viridis", norm=plot_norm_real)
    ax[1][1].set_title(l_real_label + plot_type_id)

    contour_preds_imag = ax[2][0].contourf(R, Z, g_u_imag.T, cmap="viridis", norm=plot_norm_imag)
    ax[2][0].invert_yaxis()
    ax[2][0].set_xlabel(x_label)
    ax[2][0].set_ylabel(y_label)
    ax[2][0].set_title(l_imag_pred + plot_type_id)

    contour_labels_imag = ax[2][1].contourf(R, Z, wd_imag.T, cmap="viridis", norm=plot_norm_imag)
    ax[2][1].set_title(l_imag_label + plot_type_id)

    cbar_labels_abs = fig.colorbar(contour_labels_abs, label=l_abs, ax=ax[0][1], norm=plot_norm_abs)
    cbar_preds_abs = fig.colorbar(contour_preds_abs, label=l_abs, ax=ax[0][0], norm=plot_norm_abs)
    cbar_labels_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)
    cbar_preds_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

    cbar_labels_real = fig.colorbar(contour_labels_real, label=l_real, ax=ax[1][1], norm=plot_norm_real)
    cbar_preds_real = fig.colorbar(contour_preds_real, label=l_real, ax=ax[1][0], norm=plot_norm_real)
    cbar_labels_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
    cbar_preds_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

    cbar_labels_imag = fig.colorbar(contour_labels_imag, label=l_imag, ax=ax[2][1], norm=plot_norm_imag)
    cbar_preds_imag = fig.colorbar(contour_preds_imag, label=l_imag, ax=ax[2][0], norm=plot_norm_imag)
    cbar_labels_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)
    cbar_preds_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    return fig
