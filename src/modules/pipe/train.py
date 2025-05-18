import time
from matplotlib import transforms
import yaml
import torch
import logging
import numpy as np
from typing import Any
from pathlib import Path
from .saving import Saver
from .training import TrainingLoop
from ..utilities import dir_functions
from torch.utils.data import Subset, DataLoader
from ..data_processing import data_loader as dtl
from ..model.factories.model_factory import ModelFactory
from ..data_processing.transforms import Compose, ToTensor
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..data_processing.deeponet_sampler import DeepONetSampler
from ..data_processing.deeponet_transformer import DeepONetTransformer

logger = logging.getLogger(name=__name__)

def train_model(problem_config_path: str, train_config_path: str) -> dict[str, Any]:

    # --------------------------- Load params file ------------------------
    training_params = dir_functions.load_params(file=train_config_path)
    problem_params = dir_functions.load_params(file=problem_config_path)
    dataset_path = Path(f"data/processed/{problem_params['PROBLEM']}_{problem_params['DATASET_VERSION']}")
    logger.info(msg=f"Training from:\n{dataset_path}\n")
    torch.manual_seed(seed=training_params['SEED'])

    # ---------------------------- Load dataset ----------------------

    data = dict(np.load(dataset_path / "data.npz"))
    splits = np.load(dataset_path / "split_indices.npz")
    scalers = np.load(dataset_path / "scalers.npz")
    
    with open(dataset_path / "metadata.yaml") as f:
        dataset_metadata = yaml.safe_load(f)

    to_tensor_transform = ToTensor(
        dtype=getattr(torch, training_params['PRECISION']), 
        device=training_params['DEVICE']
    )

    transforms_config = None

    # dataset_transformer = DeepONetTransformer(
    #     config=transforms_config
    # )

    train_data, val_data, test_data = dtl.slice_dataset(data=data,
                                                        feature_labels=dataset_metadata['FEATURES'],
                                                        target_labels=dataset_metadata['TARGETS'],
                                                        splits=splits
    )

    branch_label = dataset_metadata['FEATURES'][0]
    trunk_label = dataset_metadata['FEATURES'][1]
    target_labels = dataset_metadata['TARGETS']

    train_dataset = DeepONetDataset(
        data=train_data,
        feature_labels=dataset_metadata['FEATURES'],
        output_labels=target_labels,
        # transform=dataset_transformer
    )
    val_dataset = DeepONetDataset(
        data=val_data,
        feature_labels=dataset_metadata['FEATURES'],
        output_labels=target_labels,
        # transform=dataset_transformer
    )
    test_dataset = DeepONetDataset(
        data=test_data,
        feature_labels=dataset_metadata['FEATURES'],
        output_labels=target_labels,
        # transform=dataset_transformer
    )

    print(train_dataset[:]['g_u'].shape)
    print(val_dataset[:]['g_u'].shape)
    print(test_dataset[:]['g_u'].shape)

    train_branch_samples = len(train_dataset[:][branch_label])
    train_trunk_samples = len(train_dataset[:][trunk_label])

    train_sampler = DeepONetSampler(num_branch_samples=train_branch_samples,
                              branch_batch_size=training_params['BRANCH_BATCH_SIZE'],
                              num_trunk_samples=train_trunk_samples,
                              trunk_batch_size=training_params['TRUNK_BATCH_SIZE'],
                              )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=lambda x : x[0]
    )

    # ------------------------------------ Initialize model -----------------------------

    model = ModelFactory.create_model(
        model_params=training_params,
        dataset_params=dataset_metadata,
        inference=False
    )

    print(model)
    # ---------------------------------- Initializing classes for training  -------------------
    quit()
    training_loop = TrainingLoop(
        model=model,
        training_params=training_params,
    )

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()
    model_info = training_loop.train(train_batch=train_batch, val_batch=val_batch)
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")
    
    # ---------------------------- Output folder --------------------------------


    model_name = dir_functions.get_model_name(config=training_params)
    dir_functions.create_output_directories(config=problem_params)

    logger.info(msg=f"\nExperiment will be saved at:\n{problem_params['OUTPUT_PATH']}\n")

    saver = Saver(model_name=training_params['MODEL_NAME'])

    return model_info