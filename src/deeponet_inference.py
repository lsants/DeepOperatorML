import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from datetime import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

path_to_data = os.path.join(project_dir, 'data')
path_to_images = os.path.join(project_dir, 'images')
path_to_models = os.path.join(project_dir, 'models')

precision = torch.float32

def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

def get_last_model(path):
    begin = -12
    end = -4
    file_list = glob.glob(path + '/*.pth')
    last_timestamp = ''
    searched_file = ''
    for file in file_list:
        if file[begin:end] > last_timestamp:
            last_timestamp = file[begin:end]
            searched_file = file
    return searched_file

def G(data, x):
    a,b,c = data.T
    return (a)*x.T**2 + (b)*x.T + c

# ---------------- Load data -------------------
d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
y, u_test, y_test, G_test =  load_data((d['sensors'],d['X_branch'], d['X_trunk'], d['y']))
q = y.shape[0]

# ---------------- Load model -----------------
loaded_model_path = get_last_model(path_to_models)
model = torch.load(loaded_model_path)
model.eval()

# ---------------- Testing one data point ------
x = torch.tensor(np.linspace(y[0], y[-1],q))

a = 0.5 # x^2
b = 1 # x
c = 1 # 1
d = 0

f = [a,b,c]

u1 = torch.tensor(f, dtype=precision).reshape(1,-1)
G1_exact = (a/3*x**3 + b/2*x**2 + c*x + d)
G_pred = model(u1, x)

# -------------- Plots -------------------
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

ax[0].plot(x, G(u1, x).T, label='u(x) = ax^2 + x + c')
ax[0].set_xlabel('x')
ax[0].set_xlim([-10,10])
ax[0].legend()

ax[1].plot(x, G1_exact, label='G(u)(y) = (a/3)x^3 + (b/2)x^2 + cx + d')
ax[1].plot(x, G_pred.detach().numpy().T, label='model output')
ax[1].set_xlabel('x')
ax[1].set_xlim([-10,10])
ax[1].legend()

fig.suptitle('a = {}, b = {}, c = {}, d = {}'.format(a,b,c,d))
# fig.tight_layout()

plt.show()

date = datetime.today().strftime('%Y%m%d')
fig_name = f"deeponet_prediction_plots_{date}.png"

image_path = os.path.join(path_to_images, fig_name)

fig.savefig(image_path)