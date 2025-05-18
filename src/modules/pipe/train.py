import time
import yaml
import torch
import logging
import numpy as np
from typing import Any
from pathlib import Path
from torch.utils.data import Subset
from .saving import Saver
from ..utilities import dir_functions
from .training import TrainingLoop
from ..data_processing import batching as bt
from ..data_processing import data_loader as dtl
from ..deeponet.factories.model_factory import ModelFactory
from ..data_processing.transforms import Compose, ToTensor
from ..data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(name=__name__)

def train_model(problem_config_path: str, train_config_path: str) -> dict[str, Any]:

    # --------------------------- Load params file ------------------------
    training_params = dir_functions.load_params(file=train_config_path)
    problem_params = dir_functions.load_params(file=problem_config_path)
    dataset_path = Path(f"data/processed/{problem_params['PROBLEM']}_{problem_params['DATASET_VERSION']}")
    logger.info(msg=f"Training from:\n{dataset_path}\n")
    torch.manual_seed(seed=training_params['SEED'])

    # ---------------------------- Load dataset ----------------------

    data = np.load(dataset_path / "data.npz")
    splits = np.load(dataset_path / "split_indices.npz")
    scalers = np.load(dataset_path / "scalers.npz")
    
    with open(dataset_path / "metadata.yaml") as f:
        dataset_metadata = yaml.safe_load(f)

    to_tensor_transform = ToTensor(
        dtype=getattr(torch, training_params['PRECISION']), 
        device=training_params['DEVICE']
    )

    # normalization_transform = ToTensor(
    #     dtype=getattr(torch, training_params['PRECISION']), 
    #     device=training_params['DEVICE']
    # )

    transformations = Compose(transforms=[
        # normalization_transform,
        # feature_expansion_transform,
        to_tensor_transform
    ])


    full_dataset = DeepONetDataset(
        data=data,
        output_keys=problem_params['OUTPUT_KEYS'],
        transform=transformations
    )

    print([i for i in splits])

    # Use precomputed indices
    train_dataset = Subset(full_dataset, splits['XB_TRAIN'])
    val_dataset = Subset(full_dataset, splits['XB_VAL'])
    test_dataset = Subset(full_dataset, splits['XB_TEST'])

    # ------------------------------------ Initialize model -----------------------------

    model = ModelFactory.create_model(
        model_params=training_params,
        dataset_params=dataset_metadata,
        inference=False
    )


    # ---------------------------------- Initializing classes for training  -------------------


    training_strategy = model.training_strategy

    training_loop = TrainingLoop(
        model=model,
        training_params=training_params,
    )

    # ---------------------------------- Batching data -------------------------------------

    train_batch = bt.get_single_batch(dataset=train_dataset, indices=splits['XB_TRAIN'], training_params=training_params)
    val_batch = bt.get_single_batch(dataset=val_dataset, indices=splits['XB_VAL'], training_params=training_params)

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()
    model_info = training_loop.train(train_batch=train_batch, val_batch=val_batch)
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")
    
    # ---------------------------- Output folder --------------------------------

    problem_params['MODEL_NAME'] = model_name
    dir_functions.create_output_directories(config=problem_params)

    logger.info(msg=f"\nExperiment will be saved at:\n{problem_params['OUTPUT_PATH']}\n")

    saver = Saver(model_name=training_params['MODEL_NAME'])

    return model_info