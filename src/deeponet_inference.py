import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')

path_to_first_model = os.path.join(path_to_models, 'deeponet_model.pth')


def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

# ---------------- Load data -------------------
d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
x, u_test, y_test, G_test =  load_data((d['sensors'],d['X_branch'], d['X_trunk'], d['y']))

# ---------------- Load model -----------------

model = torch.load(path_to_first_model)
model.eval()

# ---------------- Testing one data point ------
N = 1000 # number of points for trapezoid method

x = x.T

u1 = torch.cos(x)
G1_exact = torch.sin(x)
G_pred = model(u1, x.T)

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(x.T, u1.T, label='u(x) = cos(x)')
ax[0].set_xlabel('x')
ax[0].legend()

ax[1].plot(x.T, G1_exact.T, label='G(u)(y) = sin(y)')
ax[1].plot(x.T, G_pred.detach().numpy().T, label='model output')
ax[1].set_xlabel('x')
ax[1].legend()

# fig.tight_layout()

plt.show()