import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
import time
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from src import generic as gnc
from src import nn_architecture as NN
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN.NeuralNetwork().to(device, dtype=torch.float32)
model.load_state_dict(torch.load(
    '/home/lsantiago/workspace/ic/project/models/model_200.pth'))


input_X = torch.tensor([1, 0, 0, 1], dtype=torch.float32).to(device)
input_X = input_X.unsqueeze(0)
input_X = gnc.normalize(input_X)
N = 1000
tests = np.zeros(N)
for i in range(N):
    with torch.no_grad():
        start = time.perf_counter_ns()
        y_pred = model(input_X)
        duration = time.perf_counter_ns() - start
        prediction = y_pred.cpu().numpy()
    tests[i] = duration
print(f"NN integration time: {np.mean(tests) / 1000 :.4f} ± {np.std(tests) / 1000:.4f} us")
print(prediction)
