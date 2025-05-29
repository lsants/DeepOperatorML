import time
import torch
import logging
import dataclasses
import numpy as np
from typing import Any
from .saving import Saver
from ..model.config import ModelConfig
from .training_loop import TrainingLoop
from torch.utils.data import DataLoader
from ..model.model_factory import ModelFactory
from ..plotting import plot_training
from ..data_processing import data_loader as dtl
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..data_processing.deeponet_sampler import DeepONetSampler
from ..data_processing.deeponet_transformer import DeepONetTransformPipeline
from ..pipe.pipeline_config import TrainConfig, DataConfig, ExperimentConfig, PathConfig

logger = logging.getLogger(name=__name__)

def get_split_data(data: Any, split_indices: dict[str, np.ndarray], features_keys: list[str], targets_keys: list[str]) -> tuple[dict[str, np.ndarray[Any, Any]], ...]:
    branch_key = features_keys[0]
    trunk_key = features_keys[1]

    train_indices = (split_indices[f'{branch_key.upper()}_train'], split_indices[f'{trunk_key.upper()}_train'])
    val_indices = (split_indices[f'{branch_key.upper()}_val'],        split_indices[f'{trunk_key.upper()}_val'])
    test_indices = (split_indices[f'{branch_key.upper()}_test'],        split_indices[f'{trunk_key.upper()}_test'])

    train_data = dtl.slice_data(
        data=data, 
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=train_indices
        )

    val_data = dtl.slice_data(
        data=data, 
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=val_indices
        )

    test_data = dtl.slice_data(
        data=data, 
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=test_indices
        )
    
    return train_data, val_data, test_data

def get_transformed_data(data: Any, features_keys: list[str], targets_keys: list[str], transform_pipeline: DeepONetTransformPipeline )-> dict[str, Any]:
    transformed_data = {
        features_keys[0]: transform_pipeline.transform_branch(data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(data[features_keys[1]]),
        **{k: transform_pipeline.transform_target(data[k]) for k in targets_keys}  # Preserve original outputs
    }

    return transformed_data

def get_stats(data: dict[str, np.ndarray], keys: list[str]) -> dict[str, Any]:
    """Compute statistics for normalization."""
    stats = {}
    for key in keys:
        check = any(key in k for k in data.keys())
        if not check:
            raise KeyError(f"Key {key} not found in data.")
        else:
            stats[key] = {
                'mean': data[f'{key}_mean'],
                'std': data[f'{key}_std'],
                'min': data[f'{key}_min'],
                'max': data[f'{key}_max'],
            }
    return stats

def get_model(): pass

def train_model(
                data_cfg: DataConfig,
                train_cfg: TrainConfig,
            ):

    torch.random.manual_seed(train_cfg.seed)
    path_cfg = PathConfig.from_data_config(data_cfg=data_cfg)

    exp_cfg = ExperimentConfig.from_dataclasses(data_cfg=data_cfg, train_cfg=train_cfg, path_cfg=path_cfg)

    stats = get_stats(data=data_cfg.scalers, 
                      keys=data_cfg.features + data_cfg.targets)

    transform_pipeline = DeepONetTransformPipeline(config=train_cfg.transforms)


    transform_pipeline.set_branch_stats(
        stats=stats[data_cfg.features[0]]
    )
    transform_pipeline.set_trunk_stats(
        stats=stats[data_cfg.features[1]]
    )
    transform_pipeline.set_target_stats(
        stats=stats[data_cfg.targets[0]]
    )

    train_data, val_data, test_data = get_split_data(data=data_cfg.data,
                                                     split_indices=data_cfg.split_indices,
                                                     features_keys=data_cfg.features,
                                                     targets_keys=data_cfg.targets
    )
    train_transformed = get_transformed_data(data=train_data, 
                                             features_keys=data_cfg.features, 
                                             targets_keys=data_cfg.targets, 
                                             transform_pipeline=transform_pipeline
    )
    val_transformed = get_transformed_data(data=val_data, 
                                           features_keys=data_cfg.features, 
                                           targets_keys=data_cfg.targets, 
                                           transform_pipeline=transform_pipeline
    )

    train_cfg.model.trunk.input_dim = train_transformed['xt'].shape[1]

    train_dataset = DeepONetDataset(
        data=train_transformed,
        feature_labels=data_cfg.features,
        output_labels=data_cfg.targets,
    )
    val_dataset = DeepONetDataset(
        data=val_transformed,
        feature_labels=data_cfg.features,
        output_labels=data_cfg.targets,
    )

    train_branch_samples = len(train_dataset[:][data_cfg.features[0]])
    train_trunk_samples = len(train_dataset[:][data_cfg.features[1]])

    val_branch_samples = len(val_dataset[:][data_cfg.features[0]])
    val_trunk_samples = len(val_dataset[:][data_cfg.features[1]])

    train_sampler = DeepONetSampler(num_branch_samples=train_branch_samples,
                              branch_batch_size=train_cfg.branch_batch_size,
                              num_trunk_samples=train_trunk_samples,
                              trunk_batch_size=train_cfg.trunk_batch_size,
    )

    val_sampler = DeepONetSampler(num_branch_samples=val_branch_samples,
                              branch_batch_size=train_cfg.branch_batch_size,
                              num_trunk_samples=val_trunk_samples,
                              trunk_batch_size=train_cfg.trunk_batch_size,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=lambda x : x[0]
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        collate_fn=lambda x : x[0]
    )

    # ------------------------------------ Initialize model & train loop -----------------------------

    model, train_strategy = ModelFactory.create_for_training(
        config=ModelConfig(
            branch=train_cfg.model.branch,
            trunk=train_cfg.model.trunk,
            output=train_cfg.model.output,
            rescaling=train_cfg.model.rescaling,
            strategy=train_cfg.model.strategy
        )
    )

    PathConfig.create_directories(path_cfg)

    saver = Saver()

    saver.save_transform_pipeline(
        file_path = path_cfg.checkpoints_path,
        transform_pipeline = transform_pipeline,
    )

    loop = TrainingLoop(
        model=model,
        strategy=train_strategy,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=train_cfg.device,
        checkpoint_dir=path_cfg.checkpoints_path,
        sampler=train_sampler
    )

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()

    loop.run(total_epochs=train_cfg.epochs)
    
    end_time = time.time()

    training_time = end_time - start_time

    history = loop.history.get_history()

    fig = plot_training(history=history, plot_config=dataclasses.asdict(train_cfg))
    

    saver.save_model_info(
        file_path=path_cfg.outputs_path / 'model_info.yaml',
            model_info=dataclasses.asdict(exp_cfg)
    )

    saver.save_plots(
        file_path=path_cfg.plots_path / 'training_history.png',
        figure=fig
    )
    
    saver.save_history(
        file_path=path_cfg.metrics_path / 'train_metrics.csv',
        history=history
    )
    logger.info(msg=f"Experiment saved at {path_cfg.outputs_path}")
    
    logger.info(msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")
