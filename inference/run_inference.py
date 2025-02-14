import time
import torch
import numpy as np
from tqdm.auto import tqdm
import logging
import sys
import matplotlib.pyplot as plt
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)
from modules.utilities import dir_functions
from modules.data_processing import preprocessing as ppr
from modules.pipe.saving import Saver
from modules.pipe.model_factory import initialize_model
# from modules.animation import animate_wave
from modules.plotting.plot_comparison import plot_field_comparison, plot_axis_comparison
from modules.plotting.plot_basis import plot_basis_function
from modules.data_processing.deeponet_dataset import GreenFuncDataset

logger = logging.getLogger(__name__)

class TestEvaluator:
    def __init__(self, model, error_norm):
        self.model = model
        self.error_norm = error_norm

    def __call__(self, g_u, pred):
        self.model.eval()
        with torch.no_grad():
            test_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return test_error.detach().numpy()

# ----------------------------- Load params file ------------------------------- 

p = dir_functions.load_params('params_test.yaml')
path_to_data = p['DATAFILE']
precision = p['PRECISION']
device = p['DEVICE']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
model_location = model_folder + f"model_state_{model_name}.pth"

logger.info(f"Model loaded from: {model_location}")
logger.info(f"Data loaded from: {path_to_data}\n")

# ----------------------------- Initialize model -----------------------------

model, config = initialize_model(p['MODEL_FOLDER'], p['MODELNAME'], p['DEVICE'], p['PRECISION'])

if config['TRAINING_STRATEGY']:
    model.training_phase = 'both'

# ---------------------------- Outputs folder --------------------------------

data_out_folder = p['OUTPUT_LOG_FOLDER'] + config['TRAINING_STRATEGY'] + '/' + config['OUTPUT_HANDLING'] +  '/' + model_name + "/"
fig_folder = p['IMAGES_FOLDER'] + config['TRAINING_STRATEGY'] + '/' + config['OUTPUT_HANDLING'] + '/' + model_name + "/"


# ------------------------- Initializing classes for test  -------------------

evaluator = TestEvaluator(model, config['ERROR_NORM'])
saver = Saver(model_name=model_name, model_folder=model_folder, data_output_folder=data_out_folder, figures_folder=fig_folder)

# ------------------------- Load dataset ----------------------
data = np.load(path_to_data)
to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, precision), device=device)
output_keys = config["OUTPUT_KEYS"]
dataset = GreenFuncDataset(data, transform=to_tensor_transform, output_keys=output_keys)

if p['INFERENCE_ON'] == 'train':
    indices_for_inference = config['TRAIN_INDICES']
if p['INFERENCE_ON'] == 'val':
    indices_for_inference = config['VAL_INDICES']
if p['INFERENCE_ON'] == 'test':
    indices_for_inference = config['TRAIN_INDICES']

inference_dataset = dataset[indices_for_inference]

xb = inference_dataset['xb']
xt = dataset.get_trunk()

ground_truth = {}
for key in output_keys:
    ground_truth[key] = inference_dataset[key]

# ---------------------- Initialize normalization functions ------------------------
xb_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['xb']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['xb']['max']
)
xt_scaler = ppr.Scaling(
    min_val=config['NORMALIZATION_PARAMETERS']['xt']['min'],
    max_val=config['NORMALIZATION_PARAMETERS']['xt']['max']
)

output_scalers = {}
for key in output_keys:
    output_scalers[key] = ppr.Scaling(
        min_val=config['NORMALIZATION_PARAMETERS'][key]['min'],
        max_val=config['NORMALIZATION_PARAMETERS'][key]['max']
    )

if config['INPUT_NORMALIZATION']:
    xb = xb_scaler.normalize(xb)
    xt = xt_scaler.normalize(xt)
if config['OUTPUT_NORMALIZATION']:
    ground_truth_norm = {key: output_scalers[key].normalize(ground_truth[key]) for key in output_keys}

if config['TRUNK_FEATURE_EXPANSION']:
    xt = ppr.trunk_feature_expansion(xt, config['TRUNK_EXPANSION_FEATURES_NUMBER'])

# --------------------------------- Evaluation ---------------------------------
start_time = time.time()
if config['TRAINING_STRATEGY'].lower() == 'two_step':
    if p["PHASE"] == 'trunk':
        preds = model(xt=xt)
    elif p["PHASE"] == 'branch':
        coefs, preds= model(xb=xb)
        preds = coefs

        if config['OUTPUT_NORMALIZATION']:
            preds_normalized = coefs
            
    else:
        preds = model(xb=xb, xt=xt)

elif config['TRAINING_STRATEGY'] == 'pod':
    preds = model(xb=xb)

else:
    preds = model(xb=xb, xt=xt)

end_time = time.time()

inference_time = end_time - start_time

# If the model returns a single tensor when one output key is defined,
# convert it into a dictionary for uniformity.
if len(output_keys) == 1 and not isinstance(preds, dict):
    preds = {output_keys[0]: preds}

if config['OUTPUT_NORMALIZATION']:
    preds_norm = {}
    for key in output_keys:
        preds_norm[key] = output_scalers[key].normalize(preds[key])
    errors_norm = {}
    for key in output_keys:
        errors_norm[key] = evaluator(ground_truth_norm[key], preds_norm[key])
    for key in output_keys:
        preds[key] = output_scalers[key].denormalize(preds_norm[key])
else:
    errors_norm = {}

errors = {}
for key in output_keys:
    errors[key] = evaluator(ground_truth[key], preds[key])
    logger.info(f"Test error for {key} (physical): {errors[key]:.2%}")
    if config['OUTPUT_NORMALIZATION']:
        logger.info(f"Test error for {key} (normalized): {errors_norm[key]:.2%}")

# ------------------ Plot ---------------------

if len(output_keys) == 2:
    preds_field = preds[output_keys[0]] + preds[output_keys[1]] * 1j
    truth_field = ground_truth[output_keys[0]] + ground_truth[output_keys[1]] * 1j
else:
    preds_field = preds[output_keys[0]]
    truth_field = ground_truth[output_keys[0]]

preds_field = ppr.reshape_outputs_to_plot_format(preds_field, xt)
truth_field = ppr.reshape_outputs_to_plot_format(truth_field, xt)

xt_plot = xt
if config['TRUNK_FEATURE_EXPANSION']:
    xt_plot = xt_plot[ : , : xt.shape[-1] // (1 + 2 * config['TRUNK_EXPANSION_FEATURES_NUMBER'])]
if config['INPUT_NORMALIZATION']:
    xt_plot = xt_scaler.denormalize(xt_plot)

r, z = ppr.trunk_to_meshgrid(xt_plot)

if p['SAMPLES_TO_PLOT'] == 'all':
    s = len(indices_for_inference)
else:
    s = min(p['SAMPLES_TO_PLOT'], len(indices_for_inference))

if p['BASIS_TO_PLOT'] == 'all':
    modes_to_plot = len(indices_for_inference)
else:
    modes_to_plot = min(p['BASIS_TO_PLOT'], config.get('N_BASIS'))

for sample in tqdm(range(s), colour='MAGENTA'):
    freq = inference_dataset['xb'][sample].item()
    if p['PLOT_FIELD']:
        fig_field = plot_field_comparison(r, z, truth_field[sample], preds_field[sample], freq)
        saver(figure=fig_field, figure_prefix=f"field_for_{freq:.2f}")
    if p['PLOT_AXIS']:
        fig_axis = plot_axis_comparison(r, z, truth_field[sample], preds_field[sample], freq)
        saver(figure=fig_axis, figure_prefix=f"axis_for_{freq:.2f}")

if p['PLOT_BASIS']:
    if config['TRAINING_STRATEGY'].lower() == 'two_step':
        trunks = [net for net in model.training_strategy.trained_trunk_list]
        basis_modes = torch.stack([net.T for net, _ in zip(trunks, range(len(trunks)))], dim=0)
    elif config['TRAINING_STRATEGY'] == 'pod':
        basis_modes = torch.transpose(model.training_strategy.pod_basis, 1, 2)
    else:
        trunks = [net(xt) for net, _ in zip(model.trunk_networks, range(len(model.trunk_networks)))]
        basis_modes = torch.stack([net.T for net, _ in zip(trunks, range(len(trunks)))], dim=0)
    basis_modes = ppr.reshape_outputs_to_plot_format(basis_modes, xt)
    if basis_modes.ndim < 4:
        basis_modes = np.expand_dims(basis_modes, axis=1)
    logger.info(f"Basis set shape: {basis_modes.shape}")
    if basis_modes.shape[0] > config.get('N_BASIS'):
        # If needed, combine or slice basis modes here.
        pass
    for i in tqdm(range(1, config.get('N_BASIS') + 1), colour='CYAN'):
        fig_mode = plot_basis_function(r, z, 
                                       basis_modes[i - 1], 
                                       index=i,
                                       basis_config=config['BASIS_CONFIG'],
                                       strategy=config['TRAINING_STRATEGY'])
        saver(figure=fig_mode, figure_prefix=f"mode_{i}")

# animate_wave(g_u.real, g_u_pred=preds.real, save_name='./video')

saver(errors=errors)
saver(time=inference_time, time_prefix="inference")