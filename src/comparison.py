import time
import os
import glob
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
from src.implemented_methods_test import trapezoid_rule, gauss_quadrature_two_points

path_to_data = os.path.join(project_dir, 'data')
path_to_images = os.path.join(project_dir, 'images')
path_to_models = os.path.join(project_dir, 'models')
precision = torch.float32

def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=precision)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

def G(data, x):
    a,b,c = data.T
    a,b,c = list(map(lambda x: x.reshape(-1,1), [a,b,c]))
    return (a)*x**2 + (b)*x + c

def compute_integral(coefs, a=0.1, b=1):
    alpha, beta, gamma = coefs[:,0], coefs[:, 1], coefs[:, 2]
    integrals = (alpha / 3) * (b**3 - a**3) + (beta / 2) * (b**2 - a**2) + \
            gamma * (b - a)
    return integrals

def get_last_timestamp(path):
    begin = -12
    end = -4
    file_list = glob.glob(path + '/*.pth')
    last_timestamp = ''
    for file in file_list:
        if file[begin:end] > last_timestamp:
            last_timestamp = file[begin:end]
    return last_timestamp

def G(data, x):
    a,b,c = data.T
    return (a)*x.T**2 + (b)*x.T + c

# ----------- Load models ------------
last_timestamp = get_last_timestamp(path_to_models)
mlp_path = path_to_models + '/' + 'MLP_model_' + last_timestamp + '.pth'
deeponet_path = path_to_models + '/' + 'deeponet_model_' + last_timestamp + '.pth'
mlp = torch.load(mlp_path)
deeponet = torch.load(deeponet_path)

mlp.eval()
deeponet.eval()

# ---------- Load data --------------
d = np.load(f"{path_to_data}/mlp_dataset_test.npz", allow_pickle=True)
X_mlp, y_mlp = load_data((d['X'], d['y']))
branch_input = X_mlp[:,:-1]
X = branch_input.detach().numpy()


# ---------- Parameters -----------
start, end = 0.1, X_mlp[:,-1].max().ceil().item()
N = len(branch_input)

x = np.linspace(start, end,N)
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1,1)
trunk_input = x_tensor
x_expanded = np.concatenate([x]*N, axis=0).reshape(N,-1)
f_x = G(X, x_expanded)

mlp_time, deeponet_time, gauss_time, trap_time = [], [], [], []
for i in range(N):
    with torch.no_grad():
        start = time.perf_counter_ns()
        yp_mlp = mlp(X_mlp[i].reshape(1,-1)) 
        end = time.perf_counter_ns()
        duration = (end - start)/1000
        mlp_time.append(duration)

        start = time.perf_counter_ns()
        yp_deeponet = deeponet(branch_input[i].reshape(1,-1), trunk_input)
        end = time.perf_counter_ns()
        duration = (end - start)/1000
        deeponet_time.append(duration)


# ---------- Computing integrals ---------
    start = time.perf_counter_ns()
    yp_gauss = gauss_quadrature_two_points(f_x[i].reshape(1,-1), start, end)
    end = time.perf_counter_ns()
    duration = (end - start)/1000
    gauss_time.append(duration)
    
    start = time.perf_counter_ns()
    yp_trap = trapezoid_rule(f_x[i].reshape(1,-1), start, end, N)
    end = time.perf_counter_ns()
    duration = (end - start)/1000
    trap_time.append(duration)

# MAKE SURE YOUR RESULTS ARE ACTUALLY THE SAME !!! (they're not)
mlp_time, deeponet_time, gauss_time, trap_time = list(map(lambda x: np.array(x), 
                                                          [mlp_time, deeponet_time, gauss_time, trap_time]))


# --------- Results -------------
print(f"Runtime for MLP: {mlp_time.mean():.3f} ±  {mlp_time.std():.3f} us")
print(f"Runtime for DeepONet: {deeponet_time.mean():.3f} ±  {deeponet_time.std():.3f} us")
print(f"Runtime for Gauss: {gauss_time.mean():.3f} ±  {gauss_time.std():.3f} us")
print(f"Runtime for Trapezoid: {trap_time.mean():.3f} ±  {trap_time.std():.3f} us")