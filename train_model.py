import os
import time
import torch
import numpy as np
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)
from modules.pipe.saving import Saver
from modules.utilities import dir_functions
from modules.pipe.training import TrainingLoop
from modules.pipe.model_factory import create_model
from modules.data_processing import preprocessing as ppr
from modules.data_processing.compose_transformations import Compose
from modules.data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)

# --------------------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
logger.info(f"Training data from:\n{p['DATAFILE']}")

torch.manual_seed(p['SEED'])

# ---------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, p['PRECISION']), device=p['DEVICE'])

transformations = Compose([
    to_tensor_transform
])

data = np.load(p['DATAFILE'], allow_pickle=True)
processed_data = ppr.preprocess_npz_data(p['DATAFILE'], 
                                         p["INPUT_FUNCTION_KEYS"], 
                                         p["COORDINATE_KEYS"],
                                         direction=p["DIRECTION"] if p["PROBLEM"] == 'kelvin' else None)
dataset = DeepONetDataset(processed_data, 
                           transformations, 
                           output_keys=p['OUTPUT_KEYS'])

n_outputs = dataset.n_outputs

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

dataset_indices = {'train': train_dataset.indices,
                   'val': val_dataset.indices,
                   'test': test_dataset.indices}

p['TRAIN_INDICES'] = train_dataset.indices
p['VAL_INDICES'] = val_dataset.indices
p['TEST_INDICES'] = test_dataset.indices

p['A_DIM'] = (p['BASIS_FUNCTIONS'], len(p['TRAIN_INDICES']))

# ------------------------------ Setup data normalization functions ------------------------

norm_params = ppr.get_minmax_norm_params(train_dataset)

xb_min, xb_max = norm_params['xb']['min'], norm_params['xb']['max']
xt_min, xt_max = norm_params['xt']['min'], norm_params['xt']['max']

normalize_branch = ppr.Scaling(min_val=xb_min, max_val=xb_max)
normalize_trunk = ppr.Scaling(min_val=xt_min, max_val=xt_max)

normalization_parameters = {
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
    }
}

for key in p['OUTPUT_KEYS']:
    key_min, key_max = norm_params[key]['min'], norm_params[key]['max']
    scaling = ppr.Scaling(min_val=key_min, max_val=key_max)
    normalization_parameters[key] = {
        "min": key_min,
        "max": key_max,
        "normalize": scaling.normalize,
        "denormalize": scaling.denormalize
    }

p["NORMALIZATION_PARAMETERS"] = normalization_parameters
# ------------------------------------ Initialize model -----------------------------

model, model_name = create_model(
    model_params=p,
    train_data=train_dataset[:]
)

# ---------------------------- Outputs folder --------------------------------

p['MODELNAME'] = model_name

logger.info(f"Data will be saved at:\n{p['OUTPUT_LOG_FOLDER']}\nFigure will be saved at:\n{p['IMAGES_FOLDER']}")

# ---------------------------------- Initializing classes for training  -------------------

saver = Saver(
    model_name=p['MODELNAME'], 
    model_folder=p['MODEL_FOLDER'], 
    data_output_folder=p['MODEL_FOLDER'], 
    figures_folder=os.path.join(p["IMAGES_FOLDER"])
)

training_strategy = model.training_strategy

training_loop = TrainingLoop(
    model=model,
    training_strategy=training_strategy,
    saver=saver,
    params=p
)

# ---------------------------------- Batching data -------------------------------------

def get_single_batch(dataset, indices):
    dtype = getattr(torch, p['PRECISION'])
    device = p['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = dataset.get_trunk()
    for key in p['OUTPUT_KEYS']:
        batch[key] = torch.stack([dataset[idx][key] for idx in indices], dim=0).to(dtype=dtype, device=device)
    return batch

train_batch = get_single_batch(dataset, train_dataset.indices)
val_batch = get_single_batch(dataset, val_dataset.indices) if p.get('VAL_PERC', 0) > 0 else None

# ----------------------------------------- Train loop ---------------------------------
start_time = time.time()

training_loop.train(train_batch, val_batch)

end_time = time.time()
training_time = end_time - start_time
logger.info(f"Training concluded in: {training_time:.2f} seconds")