import time
import torch
import numpy as np
from modules import dir_functions
from modules import preprocessing as ppr
from modules.vanilla_deeponet import VanillaDeepONet
from modules.greenfunc_dataset import GreenFuncDataset
from modules.test_evaluator import TestEvaluator
from modules.saving import Saver
from modules.plotting import plot_field_comparison, plot_axis_comparison

# ----------------------------- Load params file ------------------------
p = dir_functions.load_params('params_test.yaml')
path_to_data = p['DATAFILE']
print(f"Testing data from: {path_to_data}")

# ------------------------------ Defining training parameters and output paths ---------------
precision = eval(p['PRECISION'])
device = p['DEVICE']
error_type = p['ERROR_NORM']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']
model_location = model_folder + f"model_state_{model_name}.pth"
print(f"Testing model from: {model_location}")


# ------------------------- Load indexes and normalization params for testing set ----------------
indices = dir_functions.load_indices(p['INDICES_FILE'])
norm_params = dir_functions.load_indices(p['NORM_PARAMS_FILE'])

test_indices = indices['test']

print(f"Using indices from: {p['INDICES_FILE']}")
print(f"Using normalization parameters from: {p['NORM_PARAMS_FILE']} \n")

branch_norm_params = norm_params['branch']
trunk_norm_params = norm_params['trunk']

# ------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=precision, device=device)

data = np.load(path_to_data)
dataset = GreenFuncDataset(data, transform=to_tensor_transform)

test_dataset = dataset[test_indices]

xt = dataset.get_trunk()

xb = test_dataset['xb']
g_u_real = test_dataset['g_u_real']
g_u_imag = test_dataset['g_u_imag']

# ---------------------- Setup data normalization functions ------------------------
xb_min, xb_max = torch.tensor(branch_norm_params['min'], dtype=precision), torch.tensor(branch_norm_params['max'], dtype=precision)
xt_min, xt_max = torch.tensor(trunk_norm_params, dtype=precision)

normalize_branch, normalize_trunk = ppr.Normalize(xb_min, xb_max), ppr.Normalize(xt_min, xt_max)
xb_normalized = normalize_branch(xb)
xt_normalized = normalize_trunk(xt)

# ----------------------------- Initialize model -----------------------------
u_dim = p["BRANCH_INPUT_SIZE"]
x_dim = p["TRUNK_INPUT_SIZE"]
n_branches = p['N_BRANCHES']
hidden_B = p['BRANCH_HIDDEN_LAYERS']
hidden_T = p['TRUNK_HIDDEN_LAYERS']
G_dim = p["BASIS_FUNCTIONS"]

layers_B = [u_dim] + hidden_B + [G_dim * n_branches]
layers_T = [x_dim] + hidden_T + [G_dim]

try:
    if p['ACTIVATION_FUNCTION'].lower() == 'relu':
        activation = torch.nn.ReLU()
    elif p['ACTIVATION_FUNCTION'].lower() == 'tanh':
        activation = torch.tanh
    else:
        raise ValueError
except ValueError:
    print('Invalid activation function.')

model = VanillaDeepONet(branch_layers=layers_B,
                        trunk_layers=layers_T,
                        activation=activation).to(device, precision)

model.load_state_dict(torch.load(model_location, weights_only=True))

# ------------------------- Initializing classes for test  -------------------
evaluator = TestEvaluator(model, error_type)
saver = Saver(model_name=model_name, model_folder=model_folder, data_output_folder=data_out_folder, figures_folder=fig_folder)

# --------------------------------- Evaluation ---------------------------------
start_time = time.time()
preds_real, preds_imag = model(xb_normalized, xt_normalized)
end_time = time.time()

preds = preds_real + preds_imag * 1j
g_u = g_u_real + g_u_imag * 1j

preds = ppr.reshape_from_model(preds, xt)
g_u = ppr.reshape_from_model(g_u, xt)

test_error_real = evaluator(g_u_real, preds_real)
test_error_imag = evaluator(g_u_imag, preds_imag)

print(f"Test error for real part: {test_error_real:.2%}")
print(f"Test error for imaginary part: {test_error_imag:.2%}")

errors = {'real' : test_error_real,
          'imag' : test_error_imag}

inference_time = {'time' : (end_time - start_time)}

# ------------------------------------ Plot & Save --------------------------------

# To do: remove hardcoded index and implement animation

index = 1
freq = dataset[test_indices[index]]['xb'].item()

r, z = ppr.trunk_to_meshgrid(xt)

fig_field = plot_field_comparison(r, z, g_u[index], preds[index], freq)
fig_axis = plot_axis_comparison(r, z, g_u[index], preds[index], freq)

saver(errors=errors)
saver(time=inference_time, time_prefix="inference")
saver(figure=fig_field, figure_suffix="field")
saver(figure=fig_axis, figure_suffix="axis")
