import numpy as np

data = np.loadtxt("/Users/ls/Workspace/SSI_DeepONet/data/raw/lorentz/lorentz_data_300_trajectories.csv",
                  delimiter=',')

print(data.shape)

init_conds = np.unique(data[: , [0, 1, 2]], axis=0)
coords = np.unique(data[: , 3])
print(coords.shape)
trajectories = data[: , [-3, -2, -1]].reshape(300, -1, 3)

print(trajectories[: , :501, :].shape) # truncate to 5s

# print(np.unique(data[:, [0, 1, 2]], axis=0).shape)

# print(data.reshape(100, 3, int(100/0.01), 3).shape)