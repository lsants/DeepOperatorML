import os
import time
import torch
import logging

from modules.pipe.saving import Saver
from modules.utilities import dir_functions
from modules.pipe.training import TrainingLoop
from modules.pipe.model_factory import create_model
from modules.data_processing import preprocessing as ppr
from modules.data_processing.compose_transformations import Compose
from modules.data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)

def train_model(config_path: str):
    # --------------------------- Load params file ------------------------
    p = dir_functions.load_params(config_path)
    logger.info(f"Training data from:\n{p['DATAFILE']}\n")

    torch.manual_seed(p['SEED'])

    # ---------------------------- Load dataset ----------------------
    to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, p['PRECISION']), device=p['DEVICE'])

    transformations = Compose([
        to_tensor_transform
    ])

    processed_data = ppr.preprocess_npz_data(p['DATAFILE'], 
                                            p["INPUT_FUNCTION_KEYS"], 
                                            p["COORDINATE_KEYS"],
                                            direction=p["DIRECTION"] if p["PROBLEM"] == 'kelvin' else None)
    dataset = DeepONetDataset(processed_data, 
                            transformations, 
                            output_keys=p['OUTPUT_KEYS'])

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

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

    logger.info(f"\nData will be saved at:\n{p['MODEL_FOLDER']}\n\nFigure will be saved at:\n{p['IMAGES_FOLDER']}\n")

    # ---------------------------------- Initializing classes for training  -------------------

    saver = Saver(
        model_name=p['MODELNAME'], 
        model_folder=p['MODEL_FOLDER'], 
        data_output_folder=p['MODEL_FOLDER'], 
        figures_folder=os.path.join(p["IMAGES_FOLDER"])
    )

    saver.set_logging(False)

    training_strategy = model.training_strategy

    training_loop = TrainingLoop(
        model=model,
        training_strategy=training_strategy,
        saver=saver,
        params=p
    )

    # ---------------------------------- Batching data -------------------------------------

    train_batch = ppr.get_single_batch(dataset, train_dataset.indices, p)
    val_batch = ppr.get_single_batch(dataset, val_dataset.indices, p) if p.get('VAL_PERC', 0) > 0 else None

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()

    model_info = training_loop.train(train_batch, val_batch)

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")

    return model_info

if __name__ == "__main__":
    train_model("./configs/config_train.yaml")