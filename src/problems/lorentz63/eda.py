import numpy as np
from plot_trajectories import plot_lorenz_trajectories
from plot_basis import plot_basis_3d, plot_basis_component

# data = np.loadtxt("/Users/ls/Workspace/SSI_DeepONet/data/raw/lorentz63/lorentz_data_300_trajectories.csv",
#                   delimiter=',')

data = np.load('/Users/ls/Workspace/SSI_DeepONet/data/processed/lorentz63/4970884e/data.npz')
pod_data = np.load('/Users/ls/Workspace/SSI_DeepONet/data/processed/lorentz63/4970884e/pod.npz')

stacked_basis = pod_data['stacked_basis'].T.reshape(-1, 1, pod_data['stacked_basis'].shape[0])
split_basis = pod_data['split_basis'].T.reshape(-1, 3, pod_data['split_basis'].shape[0])

trajectories = data['g_u']
coords = {'t': data['xt']}
init_conds = data['xb']

# unique_init_conds = []
# seen_init_conds = set()
# for row in data:
#     init_cond_tuple = tuple(row[:3])
#     if init_cond_tuple not in seen_init_conds:
#         unique_init_conds.append(row[:3])
#         seen_init_conds.add(init_cond_tuple)
# init_conds = np.array(unique_init_conds)

# coords = np.unique(data[: , 3])
# trajectories = data[: , [-3, -2, -1]].reshape(300, -1, 3)
# indices = np.where(coords <= 5)
# print(trajectories[: , :501, :].shape) # truncate to 5s

# print(np.unique(data[:, [0, 1, 2]], axis=0).shape)

# print(data.reshape(100, 3, int(100/0.01), 3).shape)
import matplotlib.pyplot as plt
# for i in range(len(split_basis)):
#     fig_1 = plot_basis_component(
#         coords=coords,
#         basis=split_basis[i],
#         index=i,
#         target_labels=['$\\mathbf{r}_x$', '$\\mathbf{r}_y$', '$\\mathbf{r}_z$']
#     )
#     plt.show()
#     plt.close()

f1, f3 = plot_lorenz_trajectories(
    pred_trajectories=trajectories, 
    true_trajectories=trajectories, 
    trajectory_window=coords['t'][:, 0], 
    initial_conditions=init_conds,
    num_to_plot=12)

plt.show()