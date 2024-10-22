import numpy as np
import matplotlib.pyplot as plt

def plot_labels(r ,z, wd, freq, r_source, full=True):

    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full,z)
        u_flip = np.flip(wd[1:, :], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)
    else:
        R, Z = np.meshgrid(r,z)
        wd_full = wd

    u_abs = np.abs(wd_full.T)
    l_abs= r'|$u_z$|'
    u_real = np.real(wd_full.T)
    l_real = r'$\Re(u_z)$'
    u_imag = np.imag(wd_full.T)
    l_imag = r'$\Im(u_z)$'


    fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(16, 4))
    
    contour_abs = ax[0].contourf(R,Z,u_abs, cmap="viridis")
    ax[0].invert_yaxis()
    ax[0].set_xlabel(r'$\frac{r}{a}$', fontsize=14)
    ax[0].set_ylabel(r'$\frac{z}{a}$', fontsize=14)
    ax[0].set_title(r'$|u_{zz}|$' + r' at $a_0$' + f' = {freq:.2E} Hz and ' + f'a = {r_source} m')

    contour_real = ax[1].contourf(R,Z, u_real, cmap="viridis")
    ax[1].invert_yaxis()
    ax[1].set_xlabel(r'$\frac{r}{a}$', fontsize=14)
    ax[1].set_ylabel(r'$\frac{z}{a}$', fontsize=14)
    ax[1].set_title(r'$\Re(u_{zz})$' + r' at $a_0$' + f' = {freq:.2E} Hz and ' + f'a = {r_source} m')

    contour_imag = ax[2].contourf(R,Z, u_imag, cmap="viridis")
    ax[2].invert_yaxis()
    ax[2].set_xlabel(r'$\frac{r}{a}$', fontsize=14)
    ax[2].set_ylabel(r'$\frac{z}{a}$', fontsize=14)
    ax[2].set_title(r'$\Im(u_{zz})$' + r' at $a_0$' + f' = {freq:.2E} Hz and ' + f'a = {r_source} m')

    cbar_abs = fig.colorbar(contour_abs, label=l_abs, ax=ax[0])
    cbar_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

    cbar_real = fig.colorbar(contour_real, label=l_real, ax=ax[1])
    cbar_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

    cbar_imag = fig.colorbar(contour_imag, label=l_imag, ax=ax[2])
    cbar_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

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
    ax[1][1].set_title(r'Error for $Im(u_{z})$')
    ax[1][1].legend()

    plt.grid
    plt.show()