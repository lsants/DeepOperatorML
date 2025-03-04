import os
import time
import torch
import logging

from modules.pipe.saving import Saver
from modules.utilities import dir_functions
from modules.pipe.training import TrainingLoop
from src.modules.pipe import preprocessing as ppr
from modules.factories.model_factory import ModelFactory
from modules.data_processing.compose_transformations import Compose
from modules.data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)

def train_model(config_path: str) -> dict[str, any]:

    # --------------------------- Load params file ------------------------

    params = dir_functions.load_params(config_path)
    logger.info(f"Training data from:\n{params['DATAFILE']}\n")
    torch.manual_seed(params['SEED'])

    # ---------------------------- Load dataset ----------------------

    to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, params['PRECISION']), device=params['DEVICE'])

    transformations = Compose([
        to_tensor_transform
    ])

    processed_data = ppr.preprocess_npz_data(params['DATAFILE'], 
                                            params["INPUT_FUNCTION_KEYS"], 
                                            params["COORDINATE_KEYS"],
                                            direction=params["DIRECTION"] if params["PROBLEM"] == 'kelvin' else None)
    dataset = DeepONetDataset(processed_data, 
                            transformations, 
                            output_keys=params['OUTPUT_KEYS'])

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [params['TRAIN_PERC'], params['VAL_PERC'], params['TEST_PERC']])

    params['TRAIN_INDICES'] = train_dataset.indices
    params['VAL_INDICES'] = val_dataset.indices
    params['TEST_INDICES'] = test_dataset.indices

    params['A_DIM'] = (params['BASIS_FUNCTIONS'], len(params['TRAIN_INDICES']))

    # ------------------------------ Setup data normalization ------------------------

    params["NORMALIZATION_PARAMETERS"] = ppr.get_norm_params(train_dataset, params)

    # ------------------------------------ Initialize model -----------------------------

    model, model_name = ModelFactory.create_model(
        model_params=params,
        train_data=train_dataset[:],
        inference=False
    )

    # ---------------------------- Outputs folder --------------------------------

    params['MODELNAME'] = model_name
    logger.info(f"\nData will be saved at:\n{params['MODEL_FOLDER']}\n\nFigure will be saved at:\n{params['IMAGES_FOLDER']}\n")

    # ---------------------------------- Initializing classes for training  -------------------

    saver = Saver(
        model_name=params['MODELNAME'], 
        model_folder=params['MODEL_FOLDER'], 
        data_output_folder=params['MODEL_FOLDER'], 
        figures_folder=os.path.join(params["IMAGES_FOLDER"])
    )

    saver.set_logging(False)

    training_strategy = model.training_strategy

    training_loop = TrainingLoop(
        model=model,
        training_strategy=training_strategy,
        saver=saver,
        params=params
    )

    # ---------------------------------- Batching data -------------------------------------

    train_batch = ppr.get_single_batch(dataset, train_dataset.indices, params)
    val_batch = ppr.get_single_batch(dataset, val_dataset.indices, params) if params.get('VAL_PERC', 0) > 0 else None

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()
    model_info = training_loop.train(train_batch, val_batch)
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")

    return model_info

if __name__ == "__main__":
    train_model("./configs/config_train.yaml")