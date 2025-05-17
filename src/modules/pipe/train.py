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
    config = training_params | problem_params
    logger.info(msg=f"Training data from:\n{config['PROCESSED_DATA_PATH']}\n")
    torch.manual_seed(seed=config['SEED'])

    # ---------------------------- Load dataset ----------------------
    dataset_path = Path(f"data/processed/{config['PROBLEM']}_{config['DATASET_VERSION']}")

    data = np.load(dataset_path / "data.npz")
    splits = np.load(dataset_path / "split_indices.npz")
    scalers = np.load(dataset_path / "scalers.npz")
    
    with open(dataset_path / "metadata.yaml") as f:
        dataset_metadata = yaml.safe_load(f)

    _validate_config_consistency(train_config=config, dataset_metadata=dataset_metadata)

    to_tensor_transform = ToTensor(
        dtype=getattr(torch, config['PRECISION']), 
        device=config['DEVICE']
    )

    if config["INPUT_NORMALIZATION"] != 'none':
        input_normalization_transform = NormalizeTransform(
            scalers=scalers,
            normalization_type=config["INPUT_NORMALIZATION"]
        )
    if config["OUTPUT_NORMALIZATION"] != 'none':
        output_normalization_transform = NormalizeTransform(
            scalers=scalers,
            normalization_type=config["INPUT_NORMALIZATION"]
        )

        transformations = Compose(transforms=[
            normalization_transform,
            feature_expansion_transform,
            to_tensor_transform
        ])

    full_dataset = DeepONetDataset(
        data=data,
        output_keys=config['OUTPUT_KEYS'],
        transform=transformations
    )

    # Use precomputed indices
    train_dataset = Subset(full_dataset, splits['TRAIN'])
    val_dataset = Subset(full_dataset, splits['VAL'])
    test_dataset = Subset(full_dataset, splits['TEST'])

    # Store split information in config
    config.update({
        'TRAIN_INDICES': splits['TRAIN'].tolist(),
        'VAL_INDICES': splits['VAL'].tolist(),
        'TEST_INDICES': splits['TEST'].tolist(),
        'NORMALIZATION_PARAMETERS': {
            'xb': {
                   'min': scalers['xb_min'] , 'max': scalers['xb_max'],
                   'mean': scalers['xb_mean'], 'std': scalers['xb_std']
                   },
            'xt': {
                   'min': scalers['xt_min'] , 'max': scalers['xt_max'], 
                   'mean': scalers['xt_mean'], 'std': scalers['xt_std']
                   } # add output keys
        }
    })

    # ------------------------------------ Initialize model -----------------------------

    model, model_name = ModelFactory.create_model(
        model_params=config,
        train_data=train_dataset[:],
        inference=False
    )

    # ---------------------------- Output folder --------------------------------

    config['MODEL_NAME'] = model_name
    dir_functions.create_output_directories(config=config)

    logger.info(msg=f"\nExperiment will be saved at:\n{config['OUTPUT_PATH']}\n")

    # ---------------------------------- Initializing classes for training  -------------------

    saver = Saver(model_name=config['MODEL_NAME'])

    training_strategy = model.training_strategy

    training_loop = TrainingLoop(
        model=model,
        training_strategy=training_strategy,
        saver=saver,
        training_params=config,
    )

    # ---------------------------------- Batching data -------------------------------------

    train_batch = bt.get_single_batch(dataset=train_dataset, indices=splits['TRAIN'], training_params=config)
    val_batch = bt.get_single_batch(dataset=val_dataset, indices=splits['TRAIN'], training_params=config)

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()
    model_info = training_loop.train(train_batch=train_batch, val_batch=val_batch)
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")

    return model_info

def _validate_config_consistency(train_config: dict, dataset_metadata: dict):
    """Ensure training config matches dataset version parameters"""
    validation_fields = [
        ('PROBLEM', 'problem_name'),
        ('SPLITTING.SEED', 'split_seed'),
        ('NORMALIZATION.FEATURES', 'normalized_features')
    ]
    
    errors = []
    for config_key, metadata_key in validation_fields:
        train_val = _nested_get(train_config, config_key)
        dataset_val = dataset_metadata.get(metadata_key)
        
        if train_val != dataset_val:
            errors.append(f"Config mismatch: {config_key} ({train_val}) != "
                         f"dataset {metadata_key} ({dataset_val})")
    
    if errors:
        raise ValueError(f"Config/dataset version mismatch:\n" + "\n".join(errors))

def _nested_get(d: dict, key: str):
    """Get nested dictionary values using dot notation"""
    keys = key.split('.')
    for k in keys:
        d = d.get(k, {})
    return d