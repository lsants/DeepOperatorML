import os
import time
import torch
import logging
from .modules.pipe.saving import Saver
from .modules.utilities import dir_functions
from .modules.pipe.training import TrainingLoop
from .modules.data_processing import batching as bt
from .modules.data_processing import data_loader as dtl
from .modules.deeponet.factories.model_factory import ModelFactory
from .modules.data_processing.transforms import Compose, ToTensor
from .modules.data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)

def train_model(config_path: str) -> dict[str, any]:

    # --------------------------- Load params file ------------------------

    training_params = dir_functions.load_params(config_path)
    logger.info(f"Training data from:\n{training_params['DATA_FILE']}\n")
    torch.manual_seed(training_params['SEED'])

    # ---------------------------- Load dataset ----------------------

    to_tensor_transform = ToTensor(dtype=getattr(torch, training_params['PRECISION']), device=training_params['DEVICE'])

    transformations = Compose([
        to_tensor_transform
    ])

    processed_data = dtl.preprocess_npz_data(training_params['DATA_FILE'], 
                                             training_params["INPUT_FUNCTION_KEYS"], 
                                             training_params["COORDINATE_KEYS"],
                                             direction=training_params["DIRECTION"] if training_params["PROBLEM"] == 'kelvin' else None)
    dataset = DeepONetDataset(processed_data, 
                              transformations, 
                              output_keys=training_params['OUTPUT_KEYS'])

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [training_params['TRAIN_PERC'], training_params['VAL_PERC'], training_params['TEST_PERC']])

    training_params['TRAIN_INDICES'] = train_dataset.indices
    training_params['VAL_INDICES'] = val_dataset.indices
    training_params['TEST_INDICES'] = test_dataset.indices

    # ------------------------------ Setup data normalization ------------------------

    training_params["NORMALIZATION_PARAMETERS"] = dtl.get_norm_params(train_dataset, training_params)

    # ------------------------------------ Initialize model -----------------------------

    model, model_name = ModelFactory.create_model(
        model_params=training_params,
        train_data=train_dataset[:],
        inference=False
    )

    # ---------------------------- Outputs folder --------------------------------

    training_params['MODEL_NAME'] = model_name
    logger.info(f"\nData will be saved at:\n{training_params['MODEL_FOLDER_TO_SAVE']}\n\nFigure will be saved at:\n{training_params['IMAGES_FOLDER_TO_SAVE']}\n")

    # ---------------------------------- Initializing classes for training  -------------------

    saver = Saver(
        model_name=training_params['MODEL_NAME'], 
        model_folder=training_params['MODEL_FOLDER_TO_SAVE'], 
        data_output_folder=training_params['MODEL_FOLDER_TO_SAVE'], 
        figures_folder=os.path.join(training_params["IMAGES_FOLDER_TO_SAVE"])
    )

    saver.set_logging(False)

    training_strategy = model.training_strategy

    training_loop = TrainingLoop(
        model=model,
        training_strategy=training_strategy,
        saver=saver,
        training_params=training_params
    )


    # ---------------------------------- Batching data -------------------------------------

    train_batch = bt.get_single_batch(dataset, train_dataset.indices, training_params)
    val_batch = bt.get_single_batch(dataset, val_dataset.indices, training_params) if training_params.get('VAL_PERC', 0) > 0 else None

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()
    model_info = training_loop.train(train_batch, val_batch)
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")

    return model_info

if __name__ == "__main__":
    train_model("./configs/training/config_train.yaml")