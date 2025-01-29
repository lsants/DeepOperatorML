import time
import torch
import numpy as np
from modules.utilities import dir_functions
from modules.data_processing import preprocessing as ppr
from modules.pipe.saving import Saver
from modules.pipe.training import TrainingLoop
from modules.pipe.model_factory import create_model
from modules.data_processing.compose_transformations import Compose
from modules.data_processing.greenfunc_dataset import GreenFuncDataset

# --------------------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
print(f"Training data from: {p['DATAFILE']}")

torch.manual_seed(p['SEED'])

# ---------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, p['PRECISION']), device=p['DEVICE'])

transformations = Compose([
    to_tensor_transform
])

data = np.load(p['DATAFILE'], allow_pickle=True)
output_keys = ['g_u_real', 'g_u_imag']
dataset = GreenFuncDataset(data, transformations, output_keys=output_keys)

n_outputs = dataset.n_outputs

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

dataset_indices = {'train': train_dataset.indices,
                   'val': val_dataset.indices,
                   'test': test_dataset.indices}

p['TRAIN_INDICES'] = train_dataset.indices
p['VAL_INDICES'] = val_dataset.indices
p['TEST_INDICES'] = test_dataset.indices
p['OUTPUT_KEYS'] = output_keys

p['A_DIM'] = (p['BASIS_FUNCTIONS'], len(p['TRAIN_INDICES']))

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
    train_data=train_dataset[:]
)

# ---------------------------- Outputs folder --------------------------------

p['MODELNAME'] = model_name

data_out_folder = p['OUTPUT_LOG_FOLDER'] + p['TRAINING_STRATEGY'] + '/' + p['OUTPUT_HANDLING'] +  '/' + model_name + "/"
fig_folder = p['IMAGES_FOLDER'] + p['TRAINING_STRATEGY'] + '/' + p['OUTPUT_HANDLING'] + '/' + model_name + "/"

print(data_out_folder, fig_folder)

# ---------------------------------- Initializing classes for training  -------------------

saver = Saver(
    model_name=p['MODELNAME'], 
    model_folder=p['MODEL_FOLDER'], 
    data_output_folder=data_out_folder, 
    figures_folder=fig_folder
)

best_model_checkpoint = None

training_strategy = model.training_strategy

training_loop = TrainingLoop(
    model=model,
    training_strategy=training_strategy,
    saver=saver,
    params=p
)

def get_single_batch(dataset, indices):
    dtype = getattr(torch, p['PRECISION'])
    device = p['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = dataset.get_trunk()
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
