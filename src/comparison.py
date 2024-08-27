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
mlp_model = torch.load(mlp_path)
deeponet_model = torch.load(deeponet_path)

mlp_model.eval()
deeponet_model.eval()

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

yp_mlp = mlp_model(X_mlp) # What does this give?
yp_deeponet = deeponet_model(branch_input, trunk_input) # What does this give??? check


# ---------- Computing integrals ---------