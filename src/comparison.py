import torch
import torch.quantization
from src import nn_architecture as NN
from src import generic as gnc
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import integrate
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN.NeuralNetwork().to(device, dtype=torch.float32)
model.load_state_dict(torch.load(
    '/home/lsantiago/workspace/ic/project/models/model_200e_200b_0_00001lr.pth'))
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8)

# ------- Inputs for integration models ----------
input_X = torch.tensor([1, 1, 1, 1], dtype=torch.float32).to(device)  # NN
def f(x): return x**2 + x + 1  # Quad


x = np.linspace(0, 1, num=1000)
f_trap = x**2 + x + 1

N = 100

# -------- Neural Network ------------
input_X = input_X.unsqueeze(0)
input_X = gnc.normalize(input_X).to(dtype=torch.float32)
tests_NN = np.zeros(N)
for i in range(N):
    with torch.no_grad():
        start = time.perf_counter_ns()
        y_pred = model(input_X)
        duration = time.perf_counter_ns() - start
        prediction = y_pred.cpu().numpy().squeeze()
    tests_NN[i] = duration
print(
    f"NN integration time: {np.mean(tests_NN) / 1000 :.3f} ± {np.std(tests_NN) / 1000:.3f} us")
print(prediction)

# ---------- Fortran Quadpack --------------
tests_python = np.zeros(N)
for i in range(N):
    start = time.perf_counter_ns()
    y = integrate.quad(f, 0.1, 1)
    duration = time.perf_counter_ns() - start
    tests_python[i] = duration
print(
    f"Quadrature integration time: {np.mean(tests_python) / 1000 :.3f} ± {np.std(tests_python) / 1000:.3f} us")
print(y[0])


# ----------- Trapezoid Method -------------
tests_trap = np.zeros(N)
for i in range(N):
    start = time.perf_counter_ns()
    y = integrate.trapezoid(f_trap, x)
    duration = time.perf_counter_ns() - start
    tests_trap[i] = duration
print(
    f"Trapezoid integration time: {np.mean(tests_trap) / 1000 :.3f} ± {np.std(tests_trap) / 1000 :.3f} us")
print(y)
