import numpy as np
import matplotlib.pyplot as plt

def plot_labels(freqs, r ,z, wd, freq_index=None, plot_type='abs', full=True, points=False):

    # wd = u.transpose(1,0,2)
    wd = wd.reshape(len(r), len(z), len(freqs))

    if not freq_index:
        freq_index = 0
    wd = wd[:,:,freq_index]

    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full,z)
        u_flip = np.flip(wd[1:, :], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)

        if plot_type == 'abs':
            u_plot = np.abs(wd_full.T)
            l = r'|$u_z$|'
        elif plot_type == 'real':
            u_plot = np.real(wd_full.T)
            l = r'Re($u_z$)'
        else:
            u_plot = np.imag(wd_full.T)
            l = r'Im($u_z$)'
    
    else:
        R, Z = np.meshgrid(r,z)

        if plot_type == 'abs':
            u_plot = np.abs(wd)
            l = r'|$u_z$|'
        elif plot_type == 'real':
            u_plot = np.real(wd)
            l = r'Re($u_z$)'
        else:
            u_plot = np.imag(wd)
            l = r'Im($u_z$)'


    contour = plt.contourf(R,Z,u_plot, cmap="viridis")
    if points:
        points = plt.scatter(R.flatten(), Z.flatten(), c='white', edgecolors='black', s=10, label='Data Points')
    plt.gca().invert_yaxis()

    plt.xlabel('r')
    plt.ylabel('z')

    cbar = plt.colorbar(contour, label=l)
    cbar.ax.set_ylabel(l, rotation=270, labelpad=15)

    if plot_type == 'abs':
        plt.title(f'Absolute Displacement at ω = {freqs[freq_index]:.2f} rad/s')
    elif plot_type == 'real':
        plt.title(f'Real Displacement at ω = {freqs[freq_index]:.2f} rad/s')
    else:
        plt.title(f'Imaginary Displacement at ω = {freqs[freq_index]:.2f} rad/s')
    plt.legend()

    plt.show()

def plot_vals(u_pred, xt, g_u_pred):
    pass

def plot_training(epochs, train_loss, train_error_real, train_error_imag, test_error_real, test_error_imag):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,9))

    ax[0][0].plot(epochs, [i for i in train_loss], label='train_z')
    ax[0][0].set_xlabel('epoch')
    ax[0][0].set_ylabel('MSE')
    ax[0][0].set_yscale('log')
    ax[0][0].set_title(r'$u_{z}$ Loss')
    ax[0][0].legend()

    ax[0][1].plot(epochs, [np.sqrt(i**2 + j**2) for i,j in zip(train_error_real, train_error_imag)], label='abs_train')
    ax[0][1].plot(epochs, [np.sqrt(i**2 + j**2) for i,j in zip(test_error_real, test_error_imag)], label='abs_test')
    ax[0][1].set_xlabel('epoch')
    ax[0][1].set_ylabel(r'$L_2$ norm')
    ax[0][1].set_yscale('log')
    ax[0][1].set_title(r'Error for $|u_{z}|$')
    ax[0][1].legend()

    ax[1][0].plot(epochs, [i for i in train_error_real], label='real_train')
    ax[1][0].plot(epochs, [i for i in test_error_real], label='real_test')
    ax[1][0].set_xlabel('epoch')
    ax[1][0].set_ylabel(r'$L_2$ norm')
    ax[1][0].set_yscale('log')
    ax[1][0].set_title(r'Error for $Re(u_{z})$')
    ax[1][0].legend()

    ax[1][1].plot(epochs, [i for i in train_error_imag], label='imag_train')
    ax[1][1].plot(epochs, [i for i in test_error_imag], label='imag_test')
    ax[1][1].set_xlabel('epoch')
    ax[1][1].set_ylabel(r'$L_2$ norm')
    ax[1][1].set_yscale('log')
    ax[1][1].set_title(r'Error for $Im(u_{z})$')
    ax[1][1].legend()

    plt.grid
    fig.show()