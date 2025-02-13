import numpy as np
import matplotlib.pyplot as plt

def plot_basis_function(r, z, basis, **kwargs):
    index = kwargs.get('index')
    full = kwargs.get('full', True)
    non_dim_plot = kwargs.get('non_dim_plot', True)
    basis_config = kwargs.get('basis_config', 'single')

    mode_computed_with = kwargs.get('strategy').upper()

    if full:
        r_full = np.concatenate((-np.flip(r[1 : ]), r))
        R, Z = np.meshgrid(r_full, z)
        basis_flip = np.flip(basis[: , 1 : , : ], axis=1)
        basis_full = np.concatenate((basis_flip, basis), axis=1)
        basis_full = basis_full.transpose(0, 2, 1)
    else:
        R, Z = np.meshgrid(r,z)
        basis_full = basis.T

    if non_dim_plot:
        if index == 1:
            title = f'{index}st mode ({mode_computed_with})'
        elif index == 2:
            title = f'{index}nd mode ({mode_computed_with})'
        elif index == 3:
            title = f'{index}rd mode ({mode_computed_with})'
        else:
            title = f'{index}th mode ({mode_computed_with})'
            
        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    
    else:
        x_label = r'$r$'
        y_label = r'$z$'

    if basis_config == 'single':
        phi = basis_full[0]
        l = r'$\Phi(r,z)$'
        fig, ax = plt.subplots(nrows=1,
                        ncols=1,
                        figsize=(4, 4))
        
        contour = ax.contourf(R, Z, phi, cmap="viridis")
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title)

        cbar = fig.colorbar(contour, label=l, ax=ax)
        cbar.ax.set_ylabel(l, rotation=270, labelpad=15)

    elif basis_config == 'multiple':
        phi_real = basis_full[0]
        phi_imag = basis_full[1]
        
        l_real = r'$\Phi_{real}(r,z)$'
        l_imag = r'$\Phi_{imag}(r,z)$'

        fig, ax = plt.subplots(nrows=1,
                        ncols=2,
                        figsize=(8, 4))
        
        contour_real = ax[0].contourf(R, Z, phi_real, cmap="viridis")
        ax[0].invert_yaxis()
        ax[0].set_xlabel(x_label, fontsize=14)
        ax[0].set_ylabel(y_label, fontsize=14)
        ax[0].set_title(title + ' for real part')

        cbar_real = fig.colorbar(contour_real, label=l_real, ax=ax[0])
        cbar_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
        
        contour_imag = ax[1].contourf(R, Z, phi_imag, cmap="viridis")
        ax[1].invert_yaxis()
        ax[1].set_xlabel(x_label, fontsize=14)
        ax[1].set_ylabel(y_label, fontsize=14)
        ax[1].set_title(title + ' for imaginary part')

        cbar_imag = fig.colorbar(contour_imag, label=l_imag, ax=ax[1])
        cbar_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig