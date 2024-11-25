
def plot_basis_function(r , z, wd, index=None, full=True, non_dim_plot=True):
    if index:
        index += 1
    if full:
        r_full = np.concatenate((-np.flip(r[1 : ]), r))
        R, Z = np.meshgrid(r_full,z)
        u_flip = np.flip(wd[1 : , : ], axis=0)
        wd_full = np.concatenate((u_flip, wd), axis=0)
        wd_full = wd_full.T
    else:
        R, Z = np.meshgrid(r,z)
        wd_full = wd.T

    u = wd_full
    l = r'$\Phi(r,z)$'

    if non_dim_plot:
        if not index:
            title = f"Mode"
        elif index == 1:
            title = f'{index}st mode'
        elif index == 2:
            title = f'{index}nd mode'
        elif index == 3:
            title = f'{index}rd mode'
        else:
            title = f'{index}th mode'
            

        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    
    else:
        x_label = r'$r$'
        y_label = r'$z$'

    fig, ax = plt.subplots(nrows=1,
                       ncols=1,
                       figsize=(4, 4))
    
    print(u.shape)
    contour = ax.contourf(R, Z, u, cmap="viridis")
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title)

    cbar = fig.colorbar(contour, label=l, ax=ax)
    cbar.ax.set_ylabel(l, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig

def plot_pod_basis(r, z, wd, index=None, full=True, non_dim_plot=True):
    index += 1
    if full:
        r_full = np.concatenate((-np.flip(r[1:]), r))
        R, Z = np.meshgrid(r_full, z)
        u_flip = np.flip(wd[ : , 1 : , : ], axis=1)
        wd_full = np.concatenate((u_flip, wd), axis=1)
    else:
        R, Z = np.meshgrid(r,z)

    u_real = wd_full[0].T
    u_imag = wd_full[1].T

    l = r'$\phi(r,z)$'

    if non_dim_plot:
        if index == 1:
            title = f'{index}st POD mode'
        elif index == 2:
            title = f'{index}nd POD mode'
        elif index == 3:
            title = f'{index}rd POD mode'
        else:
            title = f'{index}th POD mode'

        x_label = r'$\frac{r}{a}$'
        y_label = r'$\frac{z}{a}$'
    
    else:
        x_label = r'$r$'
        y_label = r'$z$'

    fig, ax = plt.subplots(nrows=1,
                       ncols=2,
                       figsize=(8, 4))
    
    contour_real = ax[0].contourf(R, Z, u_real, cmap="viridis")
    ax[0].invert_yaxis()
    ax[0].set_xlabel(x_label, fontsize=14)
    ax[0].set_ylabel(y_label, fontsize=14)
    ax[0].set_title(title + ' real')

    cbar_real = fig.colorbar(contour_real, label=l, ax=ax[0])
    cbar_real.ax.set_ylabel(l, rotation=270, labelpad=15)
    
    contour_imag = ax[1].contourf(R, Z, u_imag, cmap="viridis")
    ax[1].invert_yaxis()
    ax[1].set_xlabel(x_label, fontsize=14)
    ax[1].set_ylabel(y_label, fontsize=14)
    ax[1].set_title(title + ' imag')

    cbar_imag = fig.colorbar(contour_imag, label=l, ax=ax[1])
    cbar_imag.ax.set_ylabel(l, rotation=270, labelpad=15)

    fig.tight_layout()
    return fig
