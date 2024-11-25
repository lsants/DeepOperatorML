import numpy as np
import matplotlib.pyplot as plt

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