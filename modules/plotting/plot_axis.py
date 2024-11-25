import numpy as np
import matplotlib.pyplot as plt

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
