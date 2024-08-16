import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')

path_to_first_model = os.path.join(path_to_models, 'deeponet_model.pth')

from src.deeponet_architecture import FNNDeepOnet

def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

# ---------------- Load data -------------------
d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
u_test, y_test, G_u_y_test =  load_data((d['X_branch'], d['X_trunk'], d['y']))

# ---------------- Load model -----------------

model = torch.load(path_to_first_model)
model.eval()

print(model(u_test, y_test))