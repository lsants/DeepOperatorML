import time
import torch
import yaml
import numpy as np
from tqdm.auto import tqdm
from modules import plotting
from modules import dir_functions
from modules import preprocessing as ppr
from modules.saving import Saver
from modules.model_factory import initialize_model
# from modules.animation import animate_wave
from modules.test_evaluator import TestEvaluator
from modules.greenfunc_dataset import GreenFuncDataset

# ----------------------------- Load params file ------------------------
p = dir_functions.load_params("models/model_info_20241121_deeponet_two_step_resnet_resnet_input_output_normalization.yaml")
path_to_data = p['DATAFILE']
precision = eval(p['PRECISION'])
device = p['DEVICE']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']
model_location = "models/model_info_20241121_deeponet_two_step_resnet_resnet_input_output_normalization.yaml"

print(f"Data from: {path_to_data}\n")

# ------------------------- Load dataset ----------------------
data = np.load(path_to_data)
to_tensor_transform = ppr.ToTensor(dtype=precision, device=device)
dataset = GreenFuncDataset(data, transform=to_tensor_transform)

inference_dataset = dataset[p["TRAIN_INDICES"]]

xb = inference_dataset['xb']
xt = dataset.get_trunk()
g_u_real = inference_dataset['g_u_real'].T
g_u_imag = inference_dataset['g_u_imag'].T

mean_function_real = g_u_real.mean(axis=0)
mean_function_imag = g_u_imag.mean(axis=0)

target_real = g_u_real - mean_function_real
target_imag = g_u_imag - mean_function_imag

U_r, S_r , V_r = torch.linalg.svd(target_real, full_matrices=True)
U_i, S_i , V_i = torch.linalg.svd(target_imag)

explained_variance_ratio = torch.cumsum(S_r**2, dim=0)/torch.linalg.norm(S_r)**2


n = (explained_variance_ratio < 0.90).sum() + 1

A = U_r[ : , : len(S_r)] @ torch.diag(S_r)
trunk = ppr.compute_pod_modes(g_u_real)
g_u_real = ppr.reshape_from_model(g_u_real, xt)
r, z = ppr.trunk_to_meshgrid(xt)

import matplotlib.pyplot as plt
for i in range(trunk.shape[0]):
    fig = plotting.plot_basis_function(r, z, g_u_real[i])
    plt.show()