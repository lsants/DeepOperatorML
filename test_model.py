import time
import torch
import numpy as np
from tqdm.auto import tqdm
from modules.plotting import plot_labels_axis
from modules.utilities import dir_functions
from data_processing import preprocessing as ppr
from pipe.saving import Saver
from pipe.model_factory import initialize_model
# from modules.animation import animate_wave
from modules.test_evaluator import TestEvaluator
from data_processing.greenfunc_dataset import GreenFuncDataset

# ----------------------------- Load params file ------------------------
p = dir_functions.load_params('params_test.yaml')
path_to_data = p['DATAFILE']
precision = p['PRECISION']
device = p['DEVICE']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']
model_location = model_folder + f"model_state_{model_name}.pth"

print(f"Model from: {model_location}")
print(f"Data from: {path_to_data}\n")

# ----------------------------- Initialize model -----------------------------

model, config = initialize_model(p['MODEL_FOLDER'], p['MODELNAME'], p['DEVICE'], p['PRECISION'])

if config['TRAINING_STRATEGY']:
    model.training_phase = 'both'

# ------------------------- Initializing classes for test  -------------------

evaluator = TestEvaluator(model, config['ERROR_NORM'])
saver = Saver(model_name=model_name, model_folder=model_folder, data_output_folder=data_out_folder, figures_folder=fig_folder)

# ------------------------- Load dataset ----------------------
data = np.load(path_to_data)
to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, precision), device=device)
dataset = GreenFuncDataset(data, transform=to_tensor_transform)

if p['INFERENCE_ON'] == 'train':
    indices_for_inference = config['TRAIN_INDICES']
if p['INFERENCE_ON'] == 'val':
    indices_for_inference = config['VAL_INDICES']
if p['INFERENCE_ON'] == 'test':
    indices_for_inference = config['TRAIN_INDICES']

inference_dataset = dataset[indices_for_inference]

xb = inference_dataset['xb']
xt = dataset.get_trunk()
g_u_real = inference_dataset['g_u_real']
g_u_imag = inference_dataset['g_u_imag']

# ---------------------- Initialize normalization functions ------------------------
xb_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['xb']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['xb']['max']
)
xt_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['xt']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['xt']['max']
)
g_u_real_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['g_u_real']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['g_u_real']['max']
)
g_u_imag_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['g_u_imag']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['g_u_imag']['max']
)

if config['INPUT_NORMALIZATION']:
    xb = xb_scaler.normalize(xb)
    xt = xt_scaler.normalize(xt)
if config['OUTPUT_NORMALIZATION']:
    g_u_real_normalized = g_u_real_scaler.normalize(g_u_real)
    g_u_imag_normalized = g_u_imag_scaler.normalize(g_u_imag)
if config['TRUNK_FEATURE_EXPANSION']:
    xt = ppr.trunk_feature_expansion(xt, config['TRUNK_EXPANSION_FEATURES_NUMBER'])

# --------------------------------- Evaluation ---------------------------------
start_time = time.time()
if config['TRAINING_STRATEGY'].lower() == 'two_step':
    if model.training_phase == 'trunk':
        preds_real, preds_imag = model(xt=xt)
    elif model.training_phase == 'branch':
        coefs_real, coefs_imag, preds_real, preds_imag = model(xb=xb)
        g_u_real, g_u_imag = coefs_real, coefs_imag

        if config['OUTPUT_NORMALIZATION']:
            g_u_real_normalized, g_u_imag_normalized = coefs_real, coefs_imag
            
    else:
        preds_real, preds_imag = model(xb=xb, xt=xt)

elif config['TRAINING_STRATEGY'] == 'pod':
    preds_real, preds_imag = model(xb=xb)

else:
    preds_real, preds_imag = model(xb=xb, xt=xt)

end_time = time.time()

inference_time = start_time - end_time

if config['OUTPUT_NORMALIZATION']:
    preds_real_normalized = g_u_real_scaler.normalize(preds_real)
    preds_imag_normalized = g_u_imag_scaler.normalize(preds_imag)

    test_error_real_normalized = evaluator(g_u_real_scaler.normalize(g_u_real), preds_real_normalized)
    test_error_imag_normalized = evaluator(g_u_imag_scaler.normalize(g_u_imag), preds_imag_normalized)

    preds_real = g_u_real_scaler.denormalize(preds_real_normalized)
    preds_imag = g_u_imag_scaler.denormalize(preds_imag_normalized)

test_error_real = evaluator(g_u_real, preds_real)
test_error_imag = evaluator(g_u_imag, preds_imag)

errors = {
    'real_physical': test_error_real,
    'imag_physical': test_error_imag,
}

print(f"Test error for real part (physical): {test_error_real:.2%}")
print(f"Test error for imaginary part (physical): {test_error_imag:.2%}")

if config['OUTPUT_NORMALIZATION']:
    errors['real_normalized'] = test_error_real_normalized
    errors['imag_normalized'] = test_error_imag_normalized
    print(f"Test error for real part (normalized): {test_error_real_normalized:.2%}")
    print(f"Test error for imaginary part (normalized): {test_error_imag_normalized:.2%}")

xt_plot = xt
if config['TRUNK_FEATURE_EXPANSION']:
    xt_plot = xt_plot[:, : xt.shape[-1] // (1 + 2 * config['TRUNK_EXPANSION_FEATURES_NUMBER'])]
if config['INPUT_NORMALIZATION']:
    xt_plot = xt_scaler.denormalize(xt_plot)

r, z = ppr.trunk_to_meshgrid(xt_plot)

preds = preds_real + preds_imag * 1j
preds = ppr.reshape_from_model(preds, xt_plot)

g_u = g_u_real + g_u_imag * 1j
g_u = ppr.reshape_from_model(g_u, xt_plot)

if config['TRAINING_STRATEGY'] == 'two_step':
    basis_modes = (model.trained_trunk).T
elif config['TRAINING_STRATEGY'] == 'pod':
    basis_modes = torch.transpose(model.basis, 1, 2)
    basis_modes = torch.transpose(basis_modes, 0, 1)

else:
    basis_modes = model.trunk_networks[0](xt).T
basis_modes = ppr.reshape_from_model(basis_modes, xt)

if p['SAMPLES_TO_PLOT'] == 'all':
    s = len(indices_for_inference)
else:
    s = min(p['SAMPLES_TO_PLOT'], len(indices_for_inference))

if p['BASIS_TO_PLOT'] == 'all':
    b = len(basis_modes)
else:
    b = min(p['BASIS_TO_PLOT'], len(basis_modes))

for i in tqdm(range(s), colour='MAGENTA'):
    freq = inference_dataset['xb'][i].item()
    if p['PLOT_FIELD']:
        fig_field = plot_labels_axis.plot_field_comparison(r, z, g_u[i], preds[i], freq)
        saver(figure=fig_field, figure_prefix=f"field_for_{freq:.2f}")
    if p['PLOT_AXIS']:
        fig_axis = plot_labels_axis.plot_axis_comparison(r, z, g_u[i], preds[i], freq)
        saver(figure=fig_axis, figure_prefix=f"axis_for_{freq:.2f}")

if p['PLOT_BASIS']:
    for i in tqdm(range(b), colour='CYAN'):
        if config['TRAINING_STRATEGY'] == 'pod':
            fig_mode = plot_labels_axis.plot_pod_basis(r, z, basis_modes[i], index=i)
        else:
            fig_mode = plot_labels_axis.plot_basis_function(r, z, basis_modes[i], index=i)
        
        saver(figure=fig_mode, figure_prefix=f"pod_{i + 1}th_mode")

# g_u, preds = ppr.mirror(g_u), ppr.mirror(preds)

# print(g_u[0].real.shape, preds[0].real.shape)

# animate_wave(g_u.real, g_u_pred=preds.real, save_name='./video')

saver(errors=errors)
saver(time=inference_time, time_prefix="inference")