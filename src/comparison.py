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
from src.numerical_methods import trapezoid_rule, gauss_quadrature_two_points

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
    if len(coefs[0] == 4):
        b = coefs[:,3]
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
    return x**2*a[:, np.newaxis] + x*b[:, np.newaxis] + c[:, np.newaxis]

# ----------- Parameters --------------
N = 5000 # Has to be less than mlp dataset length
lower_bound = 0.1

# ----------- Load models ------------
print("Loading models...")
last_timestamp = get_last_timestamp(path_to_models)
mlp_path = path_to_models + '/' + 'MLP_model_' + last_timestamp + '.pth'
deeponet_path = path_to_models + '/' + 'deeponet_model_' + last_timestamp + '.pth'
mlp = torch.load(mlp_path)
deeponet = torch.load(deeponet_path)
mlp.eval()
deeponet.eval()
print("Ok!")

# ---------- Load data --------------
print("Loading data...")
d = np.load(f"{path_to_data}/mlp_dataset_test.npz", allow_pickle=True)
X_mlp, y_mlp = load_data((d['X'], d['y']))
X_mlp, y_mlp = X_mlp[:N], y_mlp[:N]
branch_input = X_mlp[:,:-1]
X = branch_input.detach().numpy()
upper_bound = X_mlp[:,-1]
lower_bound, upper_bound = torch.tensor(lower_bound, dtype=precision).reshape(1,1).expand(1, len(upper_bound)), upper_bound.reshape(1,-1)
print("Ok!")

#----------- Inputs --------------
bounds = torch.cat([lower_bound, upper_bound], axis=0).T # inference only on extremes of domain
trunk_input = bounds
lower_bound, upper_bound = lower_bound.flatten().detach().numpy(), upper_bound.flatten().detach().numpy()
x = np.array([np.linspace(a, b, N) for a,b in zip(lower_bound, upper_bound)])
f_x = G(X, x)

mlp_results, deeponet_results, gauss_results, trap_results = [], [], [], []
mlp_time, deeponet_time, gauss_time, trap_time = [], [], [], []

print("Computing predictions...")
# -------------------- Inference ---------------
for i in tqdm(range(N), colour='GREEN'):
    # ---------- Timing the numerical methods ---------
    data_point_num = f_x[i].reshape(1,-1)
    start = time.perf_counter_ns()
    yp_trap = trapezoid_rule(data_point_num, lower_bound[i], upper_bound[i], N)
    end = time.perf_counter_ns()
    duration = (end - start)/1000
    trap_results.append(yp_trap.item())
    trap_time.append(duration)
    
    start = time.perf_counter_ns()
    yp_gauss = gauss_quadrature_two_points(data_point_num, lower_bound[i], upper_bound[i])
    end = time.perf_counter_ns()
    duration = (end - start)/1000
    gauss_results.append(yp_gauss.item())
    gauss_time.append(duration)
    
    # --------------- Timing the deep learning models --------
    with torch.no_grad():
        data_point_mlp = X_mlp[i].reshape(1,-1)
        start = time.perf_counter_ns()
        yp_mlp = mlp(data_point_mlp) 
        end = time.perf_counter_ns()
        duration = (end - start)/1000
        mlp_results.append(yp_mlp.item())
        mlp_time.append(duration)

        data_point_deeponet = branch_input[i].reshape(1,-1), trunk_input[i].reshape(-1,1)
        start = time.perf_counter_ns()
        yp_deeponet = deeponet(*data_point_deeponet)
        end = time.perf_counter_ns()
        duration = (end - start)/1000
        integral_don = (yp_deeponet[0][-1] - yp_deeponet[0][0]).item()
        deeponet_results.append(integral_don)
        deeponet_time.append(duration)

print("Done!")

mlp_time, deeponet_time, gauss_time, trap_time = list(map(lambda x: np.array(x), 
                                                          [mlp_time, deeponet_time, gauss_time, trap_time]))
mlp_results, deeponet_results, gauss_results, trap_results = list(map(lambda x: np.array(x), 
                                                          [mlp_results, deeponet_results, gauss_results, trap_results]))

# ----------------- Results ----------------------
print("------------- Runtimes ---------------")
print(f"Runtime for MLP: {mlp_time.mean():.2f} ±  {mlp_time.std():.2f} us")
print(f"Runtime for DeepONet: {deeponet_time.mean():.2f} ±  {deeponet_time.std():.2f} us")
print(f"Runtime for Gauss: {gauss_time.mean():.2f} ±  {gauss_time.std():.2f} us")
print(f"Runtime for Trapezoid: {trap_time.mean():.2f} ±  {trap_time.std():.2f} us\n")

y_sol = compute_integral(X_mlp).detach().numpy()

print("------------- Results ---------------")
print(f"L2 error [%] for MLP: {np.linalg.norm((mlp_results - y_sol)/np.linalg.norm(y_sol)):.2%}")
print(f"L2 error [%] for DeepONet: {np.linalg.norm((deeponet_results - y_sol)/np.linalg.norm(y_sol)):.2%}")
print(f"L2 error [%] for Gauss: {np.linalg.norm((gauss_results - y_sol)/np.linalg.norm(y_sol)):.2%}")
print(f"L2 error [%] for Trapezoid: {np.linalg.norm((trap_results - y_sol)/np.linalg.norm(y_sol)):.2%}")