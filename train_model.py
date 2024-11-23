# train_model.py

import time
import copy
import torch
import numpy as np
from tqdm.auto import tqdm
from modules import dir_functions
from modules import preprocessing as ppr
from modules.saving import Saver
from modules.training import TrainingLoop
from modules.plotting import plot_training
from modules.model_factory import create_model
from modules.train_evaluator import TrainEvaluator
from modules.compose_transformations import Compose
from modules.greenfunc_dataset import GreenFuncDataset

# --------------------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
print(f"Training data from: {p['DATAFILE']}")

torch.manual_seed(p['SEED'])

# ---------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, p['PRECISION']), device=p['DEVICE'])

transformations = Compose([
    to_tensor_transform
])

data = np.load(p['DATAFILE'], allow_pickle=True)  # Ensure correct loading
output_keys = ['g_u_real', 'g_u_imag']
dataset = GreenFuncDataset(data, transformations, output_keys=output_keys)

# Get outputs count from dataset
n_outputs = dataset.n_outputs  # Ensure 'n_outputs' is defined in GreenFuncDataset

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

dataset_indices = {'train': train_dataset.indices,
                   'val': val_dataset.indices,
                   'test': test_dataset.indices}

p['TRAIN_INDICES'] = train_dataset.indices
p['VAL_INDICES'] = val_dataset.indices
p['TEST_INDICES'] = test_dataset.indices

p['A_DIM'] = (p['BASIS_FUNCTIONS'], len(p['TRAIN_INDICES']))

if p['TRAINING_STRATEGY'] == 'pod':
    pod_basis, mean_functions = ppr.get_pod_parameters(
        train_dataset, num_modes=p.get('NUM_MODES', 20)
    )

# ------------------------------ Setup data normalization functions ------------------------

norm_params = ppr.get_minmax_norm_params(train_dataset)

xb_min, xb_max = norm_params['xb']['min'], norm_params['xb']['max']
xt_min, xt_max = norm_params['xt']['min'], norm_params['xt']['max']
g_u_real_min, g_u_real_max = norm_params['g_u_real']['min'], norm_params['g_u_real']['max']
g_u_imag_min, g_u_imag_max = norm_params['g_u_imag']['min'], norm_params['g_u_imag']['max']

normalize_branch = ppr.Scaling(min_val=xb_min, max_val=xb_max)
normalize_trunk = ppr.Scaling(min_val=xt_min, max_val=xt_max)
normalize_g_u_real = ppr.Scaling(min_val=g_u_real_min, max_val=g_u_real_max)
normalize_g_u_imag = ppr.Scaling(min_val=g_u_imag_min, max_val=g_u_imag_max)

# Store normalization parameters in p["NORMALIZATION_PARAMETERS"]
p["NORMALIZATION_PARAMETERS"] = {
    "xb": {
        "min": xb_min,
        "max": xb_max,
        "normalize": normalize_branch.normalize,
        "denormalize": normalize_branch.denormalize
    },
    "xt": {
        "min": xt_min,
        "max": xt_max,
        "normalize": normalize_trunk.normalize,
        "denormalize": normalize_trunk.denormalize
    },
    "g_u_real": {
        "min": g_u_real_min,
        "max": g_u_real_max,
        "normalize": normalize_g_u_real.normalize,
        "denormalize": normalize_g_u_real.denormalize
    },
    "g_u_imag": {
        "min": g_u_imag_min,
        "max": g_u_imag_max,
        "normalize": normalize_g_u_imag.normalize,
        "denormalize": normalize_g_u_imag.denormalize
    }
}
# ------------------------------------ Initialize model -----------------------------

model, model_name = create_model(
    model_params=p,
    pod_basis=p.get('pod_basis'),
    mean_functions=p.get('mean_functions'),
    A_dim=p.get('A_DIM')
)

p['MODELNAME'] = model_name
# print(model)

# ---------------------------------- Initializing classes for training  -------------------
evaluator = TrainEvaluator(p['ERROR_NORM'])

saver = Saver(
    model_name=p['MODELNAME'], 
    model_folder=p['MODEL_FOLDER'], 
    data_output_folder=p['OUTPUT_LOG_FOLDER'], 
    figures_folder=p['IMAGES_FOLDER']
)

best_model_checkpoint = None

training_strategy = model.training_strategy

training_loop = TrainingLoop(
    model=model,
    training_strategy=training_strategy,
    evaluator=evaluator,
    saver=saver,
    params=p
)

def get_single_batch(dataset, indices):
    dtype = getattr(torch, p['PRECISION'])
    device = p['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = torch.tensor(dataset.get_trunk(), dtype=dtype, device=device)
    batch['g_u_real'] = torch.stack([dataset[idx]['g_u_real'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['g_u_imag'] = torch.stack([dataset[idx]['g_u_imag'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    return batch

train_batch = get_single_batch(dataset, train_dataset.indices)
val_batch = get_single_batch(dataset, val_dataset.indices) if p.get('VAL_PERC', 0) > 0 else None

# ----------------------------------------- Train loop ---------------------------------
start_time = time.time()

training_loop.train(train_batch, val_batch)

end_time = time.time()
training_time = end_time - start_time
print(f"Training concluded in: {training_time:.2f} seconds")
