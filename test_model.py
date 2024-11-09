import time
import torch
import numpy as np
from modules import dir_functions
from modules import preprocessing as ppr
from modules.saving import Saver
from modules.test_evaluator import TestEvaluator
from modules.vanilla_deeponet import VanillaDeepONet
from modules.greenfunc_dataset import GreenFuncDataset
from modules.plotting import plot_field_comparison, plot_axis_comparison

# ----------------------------- Load params file ------------------------
p = dir_functions.load_params('params_test.yaml')
path_to_data = p['DATAFILE']
precision = eval(p['PRECISION'])
device = p['DEVICE']
error_type = p['ERROR_NORM']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']
model_location = model_folder + f"model_state_{model_name}.pth"

print(f"Model from: {model_location}")
print(f"Data from: {path_to_data}")

# ------------------------- Load indexes and normalization params ----------------
indices = dir_functions.load_data_info(p['INDICES_FILE'])
norm_params = dir_functions.load_data_info(p['NORM_PARAMS_FILE'])

test_indices = indices['test']

print(f"Indices from: {p['INDICES_FILE']}")
print(f"Normalization parameters from: {p['NORM_PARAMS_FILE']} \n")

branch_norm_params = norm_params['xb']
trunk_norm_params = norm_params['xt']
real_part_norm_params = norm_params['g_u_real']
imag_part_norm_params = norm_params['g_u_imag']

# ------------------------- Load dataset ----------------------
data = np.load(path_to_data)

to_tensor_transform = ppr.ToTensor(dtype=precision, device=device)
dataset = GreenFuncDataset(data, transform=to_tensor_transform)
test_dataset = dataset[test_indices]

xb = test_dataset['xb']
xt = dataset.get_trunk()
g_u_real = test_dataset['g_u_real']
g_u_imag = test_dataset['g_u_imag']

# ---------------------- Initialize normalization functions ------------------------
xb_min, xb_max = torch.tensor(branch_norm_params['min'], dtype=precision), torch.tensor(branch_norm_params['max'], dtype=precision)
xt_min, xt_max = torch.tensor(trunk_norm_params['min'], dtype=precision), torch.tensor(trunk_norm_params['max'], dtype=precision)
g_u_real_min, g_u_real_max = torch.tensor(real_part_norm_params['min'], dtype=precision), torch.tensor(real_part_norm_params['max'], dtype=precision)
g_u_imag_min, g_u_imag_max = torch.tensor(imag_part_norm_params['min'], dtype=precision), torch.tensor(imag_part_norm_params['max'], dtype=precision)

normalize_branch, normalize_trunk = ppr.Normalize(xb_min, xb_max), ppr.Normalize(xt_min, xt_max)
normalize_g_u_real = ppr.Normalize(g_u_real_min, g_u_real_max)
normalize_g_u_imag = ppr.Normalize(g_u_imag_min, g_u_imag_max)

denormalize_xb = ppr.Denormalize(xb_min, xb_max)
denormalize_xt = ppr.Denormalize(xt_min, xt_max)
denormalize_g_u_real = ppr.Denormalize(g_u_real_min, g_u_real_max)
denormalize_g_u_imag = ppr.Denormalize(g_u_imag_min, g_u_imag_max)

if p['INPUT_NORMALIZATION']:
    xb = normalize_branch(xb)
    xt = normalize_trunk(xt)
if p['OUTPUT_NORMALIZATION']:
    g_u_real_normalized = normalize_g_u_real(g_u_real)
    g_u_imag_normalized = normalize_g_u_imag(g_u_imag)

if p['TRUNK_FEATURE_EXPANSION']:
    xt = ppr.trunk_feature_expansion(xt, p['EXPANSION_FEATURES_NUMBER'])

# ----------------------------- Initialize model -----------------------------
expansion_dim = p['EXPANSION_FEATURES_NUMBER']
u_dim = p["BRANCH_INPUT_SIZE"]
x_dim = p["TRUNK_INPUT_SIZE"]
n_branches = p['N_BRANCHES']
hidden_B = p['BRANCH_HIDDEN_LAYERS']
hidden_T = p['TRUNK_HIDDEN_LAYERS']
G_dim = p["BASIS_FUNCTIONS"]

if p['TRUNK_FEATURE_EXPANSION']: # 2 here is hardcoded because we add a sin(x) and cos(x) term. See if this can be improved.
    x_dim += 2 * x_dim * expansion_dim

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
preds_real, preds_imag = model(xb, xt)
end_time = time.time()


if p['OUTPUT_NORMALIZATION']:
    preds_real_normalized, preds_imag_normalized = preds_real, preds_imag
    preds_real, preds_imag = denormalize_g_u_real(preds_real_normalized), denormalize_g_u_imag(preds_imag_normalized) 
    test_error_real_normalized = evaluator(g_u_real_normalized, preds_real_normalized)
    test_error_imag_normalized = evaluator(g_u_imag_normalized, preds_imag_normalized)

test_error_real = evaluator(g_u_real, preds_real)
test_error_imag = evaluator(g_u_imag, preds_imag)


errors = {'real_physical' : test_error_real,
          'imag_physical' : test_error_imag
          }


print(f"Test error for real part (physical): {test_error_real:.2%}")
print(f"Test error for imaginary part (physical): {test_error_imag:.2%}")

if p['OUTPUT_NORMALIZATION']:
    errors['real_normalized'] = test_error_real_normalized
    errors['imag_normalized'] = test_error_imag_normalized
    print(f"Test error for real part (normalized): {test_error_real_normalized:.2%}")
    print(f"Test error for imaginary part (normalized): {test_error_imag_normalized:.2%}")

inference_time = {'time' : (end_time - start_time)}

# ------------------------------------ Plot & Save --------------------------------
# To do: remove hardcoded index and implement animation

index = 0
freq = dataset[test_indices[index]]['xb'].item()

if p['INPUT_NORMALIZATION']:
    if p['TRUNK_FEATURE_EXPANSION']:
        xt_plot = xt[ : , : p["TRUNK_INPUT_SIZE"]]
    xt_plot = denormalize_xt(xt_plot)

r, z = ppr.trunk_to_meshgrid(xt_plot)

preds = preds_real + preds_imag * 1j
preds = ppr.reshape_from_model(preds, xt_plot)

g_u = g_u_real + g_u_imag * 1j
g_u = ppr.reshape_from_model(g_u, xt_plot)

fig_field = plot_field_comparison(r, z, g_u[index], preds[index], freq)
fig_axis = plot_axis_comparison(r, z, g_u[index], preds[index], freq)

saver(errors=errors)
saver(time=inference_time, time_prefix="inference")
saver(figure=fig_field, figure_suffix="field")
saver(figure=fig_axis, figure_suffix="axis")