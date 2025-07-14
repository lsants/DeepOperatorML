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


pod_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/rajapakse_fixed_material/e8d98aa8/pod.npz'
data_path = '/Users/ls/Workspace/SSI_DeepONet/data/raw/rajapakse_fixed_material/rajapakse_fixed_v1.npz'
processed_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/rajapakse_fixed_material/e8d98aa8/data.npz'
pod_data = np.load(pod_path)
data = np.load(processed_path)
g_u = data['g_u']
single_basis = pod_data['single_basis']
single_mean = pod_data['single_mean']
multi_basis = pod_data['multi_basis']
multi_mean = pod_data['multi_mean']
print(single_basis.shape, single_mean.shape,
      multi_basis.shape, multi_mean.shape)
test = g_u.reshape(30, 20, 20, 2).transpose(0, 3, 2, 1)
print(test.shape)
fig = plot_basis_function(test[0][1])
plt.show()

test_1 = multi_mean
for i in range(test_1.shape[0]):
    fig = plot_basis_function(test_1[i].reshape(20, 20).T)
    plt.show()
# fig = plot_basis_function(g_u_full[0].T.imag)
# plt.show()
