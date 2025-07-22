import numpy as np
import matplotlib.pyplot as plt


def plot_basis_function(arr):
    arr_flip = np.flip(arr, axis=1)
    arr_full = np.concatenate((arr_flip, arr), axis=1)
    fig, axs = plt.subplots()
    axs.invert_yaxis()
    contour = axs.contourf(arr_full, cmap="viridis")
    fig.colorbar(contour, ax=axs)
    return fig


def plot_basis_function_both(basis, flipped):
    fig, axs = plt.subplots(ncols=2)
    axs[0].contourf(flipped, cmap="viridis")
    axs[1].contourf(basis, cmap="viridis")
    return fig


pod_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/rajapakse_fixed_material/a37f8126/pod.npz'
data_path = '/Users/ls/Workspace/SSI_DeepONet/data/raw/rajapakse_fixed_material/rajapakse_fixed_v2.npz'
processed_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/rajapakse_fixed_material/a37f8126/data.npz'
pod_data = np.load(pod_path)
data = np.load(processed_path)
g_u = data['g_u']
stacked_basis = pod_data['stacked_basis']
stacked_mean = pod_data['stacked_mean']
split_basis = pod_data['split_basis']
split_mean = pod_data['split_mean']
print(stacked_basis.shape, stacked_mean.shape,
      split_basis.shape, split_mean.shape)
# test = g_u.reshape(30, 20, 20, 2).transpose(0, 3, 2, 1)
# print(test.shape)
# fig = plot_basis_function(test[0][1])
# plt.show()

test_1 = split_basis.T
for i in range(test_1.shape[0]):
    print(test_1.shape)
    fig = plot_basis_function(test_1[i].reshape(40, 40).T)
    plt.show()
# fig = plot_basis_function(g_u_full[0].T.imag)
# plt.show()
