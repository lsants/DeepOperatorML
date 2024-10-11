import yaml
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

with open('data_generation_params.yaml') as file:
    p = yaml.safe_load(file)

load = 6006663.0

# Load the MATLAB data
path = '/users/lsantia9/Documents/base script 160321/influencia_data.mat'
data = sio.loadmat(path)

plot = input()

# Extract variables and remove singleton dimensions
omega = np.squeeze(data['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data['r_campo'])          # Shape: (n,)
z = np.squeeze(data['z_campo'])          # Shape: (n,)
u = load*data['wd'].transpose(0,1,2)                           # Shape: (n, n, num_freqs)


# Print shapes for debugging
print('omega shape:', omega.shape)
print('r shape:', r.shape)
print('z shape:', z.shape)
print('u shape:', u.shape)

r_campo_full = np.concatenate((-np.flip(r[1:]), r))

R, Z = np.meshgrid(r_campo_full,z)

freq_index = 7

u1 = u[:,:,freq_index]

u_flip = np.flip(u1[1:, :], axis=0)
wd_full = np.concatenate((u_flip, u1), axis=0)

if plot == 'abs':
    u_plot = np.abs(wd_full.T)
    l = r'|$u_z$|'
elif plot == 'real':
    u_plot = np.real(wd_full.T)
    l = r'Re($u_z$)'
else:
    u_plot = np.imag(wd_full.T)
    l = r'Im($u_z$)'


contour = plt.contourf(R,Z,u_plot, cmap="viridis")
# points = plt.scatter(R.flatten(), Z.flatten(), c='white', edgecolors='black', s=25, label='Data Points')
plt.gca().invert_yaxis()

# Labels and title
plt.xlabel('r')
plt.ylabel('z')

cbar = plt.colorbar(contour, label=l)
cbar.ax.set_ylabel(l, rotation=270, labelpad=15)
if plot == 'abs':
    plt.title(f'Absolute Displacement at ω = {omega[freq_index]:.2f} rad/s')
elif plot == 'real':
    plt.title(f'Real Displacement at ω = {omega[freq_index]:.2f} rad/s')
else:
    plt.title(f'Imaginary Displacement at ω = {omega[freq_index]:.2f} rad/s')
plt.legend()

plt.show()