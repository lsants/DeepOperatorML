import numpy as np
from plot_trajectories import plot_lorenz_trajectories

data = np.loadtxt("/Users/ls/Workspace/SSI_DeepONet/data/raw/lorentz63/lorentz_data_300_trajectories.csv",
                  delimiter=',')

unique_init_conds = []
seen_init_conds = set()
for row in data:
    init_cond_tuple = tuple(row[:3])
    if init_cond_tuple not in seen_init_conds:
        unique_init_conds.append(row[:3])
        seen_init_conds.add(init_cond_tuple)
init_conds = np.array(unique_init_conds)

coords = np.unique(data[: , 3])
trajectories = data[: , [-3, -2, -1]].reshape(300, -1, 3)
indices = np.where(coords <= 5)
print(trajectories[: , :501, :].shape) # truncate to 5s

# print(np.unique(data[:, [0, 1, 2]], axis=0).shape)

# print(data.reshape(100, 3, int(100/0.01), 3).shape)

plot_lorenz_trajectories(
    trajectories=trajectories, 
    trajectory_window=coords, 
    initial_conditions=init_conds,
    num_to_plot=3)