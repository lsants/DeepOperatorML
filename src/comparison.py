import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
import time
import numpy as np
from src.trapezoid_rule_implementation import trapezoid_rule
from src.gaussian_quadrature_implementation import two_point_gaussian_quadrature
from src import generate_poly_dataset as gen
from src import generic as gnc
from src import nn_architecture as NN
import torch
import matplotlib.pyplot as plt

# Generate dataset for test:
sample_size = 10000
np.random.seed(42)
X, y = gen.generate_data(sample_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN.NeuralNetwork().to(device, dtype=torch.float32)
model.load_state_dict(torch.load(
    '/home/lsantiago/workspace/ic/project/models/model_200e_200b_0_00001lr.pth'))

# _______ Print number of model parameters -------
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'Neural network has a total of {params} parameters')


# ------- Inputs for integration models ----------
input_X = torch.tensor(X, dtype=torch.float32).to(device)  # NN
a = 0.1

poly_coefs = input_X[:-1].tolist()

# -------- Neural Network ------------
input_X = input_X.unsqueeze(0)
times_NN = []
preds_NN = []
times_gauss = []
preds_gauss = []
times_trap = []
preds_trap = []
check = []
N_points = 5  # for trapezoid rule

for i in input_X[0]:
    integral = gnc.compute_integral(i.tolist())
    check.append(integral)
    global alfa, beta, gamma, b
    alfa, beta, gamma, b = i
    # Polynomial

    def f(x):
        return alfa*x**2 + beta*x + gamma
    # Neural Network
    i_norm = gnc.normalize(i).to(dtype=torch.float32)
    i_nn = torch.atleast_2d(i_norm)
    with torch.no_grad():
        start = time.perf_counter_ns()
        y_pred = model(i_nn)
        end = time.perf_counter_ns()
        duration = end - start
        prediction = y_pred.cpu().numpy().squeeze()
    times_NN.append(duration)
    preds_NN.append(y_pred.item() - integral)

    # Implemented 2-pt Gaussian quadrature
    start = time.perf_counter_ns()
    y_gauss = two_point_gaussian_quadrature(f, a, b)
    end = time.perf_counter_ns()
    duration = end - start
    times_gauss.append(duration)
    preds_gauss.append(y_gauss - integral)

    # Implemented Trapezoid rule
    start = time.perf_counter_ns()
    y_trap = trapezoid_rule(f, a, b, N_points)
    end = time.perf_counter_ns()
    duration = end - start
    times_trap.append(duration)
    preds_trap.append(y_trap - integral)

check = np.array(check)
times_NN = np.array(times_NN)
preds_NN = np.array(preds_NN)
times_gauss = np.array(times_gauss)
preds_gauss = np.array(preds_gauss)
times_trap = np.array(times_trap)
preds_trap = np.array(preds_trap)
print(
    f"NN integration time: {np.mean(times_NN)/1e3 :.3f} ± {np.std(times_NN)/1e3:.3f} us")
print(f"L1 error norm for NN: {np.mean(abs(preds_NN - check)): .3f}")
print("----------------------------------------------------")
print(
    f"Quadrature (2 points) integration time: {np.mean(times_gauss)/1e3 :.3f} ± {np.std(times_gauss)/1e3:.3f} us")
print(f"L1 error norm for Gaussian quadrature: {np.mean(abs(preds_gauss - check)): .3f}")
print("----------------------------------------------------")
print(
    f"Trapezoid ({N_points} points) integration time: {np.mean(times_trap)/1e3 :.3f} ± {np.std(times_trap)/1e3 :.3f} us")
print(f"L1 error norm for Trapezoid: {np.mean(abs(preds_trap - check)): .3f}")
print("----------------------------------------------------")

plt.hist(times_NN/1e3, bins=300)
plt.xlabel("Time [us]")
plt.title(f"Inference time for NN (Sample size = {sample_size})")
plt.show()