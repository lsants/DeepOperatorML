import time
import logging
from matplotlib.transforms import Transform
import numpy as np
from typing import Any
from .saving import Saver
from ..model.config import ModelConfig
from torch.utils.data import DataLoader
# from .training_loop import TrainingLoop
from ..model.model_factory import ModelFactory
from ..data_processing import data_loader as dtl
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..data_processing.deeponet_sampler import DeepONetSampler
from ..data_processing.deeponet_transformer import DeepONetTransformPipeline
from ..pipe.pipeline_config import TrainConfig, DataConfig, ExperimentConfig, PathConfig

logger = logging.getLogger(name=__name__)

def get_split_data(data: Any, split_indices: dict[str, np.ndarray], features_keys: list[str], targets_keys: list[str]) -> tuple[dict[str, np.ndarray[Any, Any]], ...]:
    branch_key = features_keys[0]
    trunk_key = features_keys[1]

    train_indices = (split_indices[f'{branch_key}_train'.upper()], split_indices[f'{trunk_key}_train'.upper()])
    val_indices = (split_indices[f'{branch_key}_val'.upper()],        split_indices[f'{trunk_key}_val'.upper()])
    test_indices = (split_indices[f'{branch_key}_test'.upper()],        split_indices[f'{trunk_key}_test'.upper()])

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
        **{k: data[k] for k in targets_keys}  # Preserve original outputs
    }

    return transformed_data

def get_model(): pass

def train_model(
                data_cfg: DataConfig,
                train_cfg: TrainConfig,
            ):

    experiment_cfg = ExperimentConfig.from_dataclasses(
        data_cfg=data_cfg,
        train_cfg=train_cfg
    )

    path_cfg = PathConfig.from_data_config(data_cfg=data_cfg)
    PathConfig.create_directories(path_cfg)

    saver = Saver()

    transform_pipeline = DeepONetTransformPipeline(train_cfg.transforms)
    saver.save_transform_pipeline(
        file_path = path_cfg.checkpoints_path,
        transform_pipeline = transform_pipeline,
    )

    train_data, val_data, test_data = get_split_data(data=data_cfg.data,
                                                     split_indices=data_cfg.split_indices,
                                                     features_keys=data_cfg.features,
                                                     targets_keys=data_cfg.targets)
    train_transformed = get_transformed_data(data=train_data, 
                                             features_keys=data_cfg.features, 
                                             targets_keys=data_cfg.targets, 
                                             transform_pipeline=transform_pipeline)
    val_transformed = get_transformed_data(data=val_data, 
                                           features_keys=data_cfg.features, 
                                           targets_keys=data_cfg.targets, 
                                           transform_pipeline=transform_pipeline)
    test_transformed = get_transformed_data(data=test_data, features_keys=data_cfg.features, targets_keys=data_cfg.targets, transform_pipeline=transform_pipeline)
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

    train_sampler = DeepONetSampler(num_branch_samples=train_branch_samples,
                              branch_batch_size=train_cfg.branch_batch_size,
                              num_trunk_samples=train_trunk_samples,
                              trunk_batch_size=train_cfg.trunk_batch_size,
                              )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=lambda x : x[0]
    )
    # ------------------------------------ Initialize model -----------------------------

    model = ModelFactory.create_for_training(
        config=ModelConfig(
            branch=train_cfg.model.branch,
            trunk=train_cfg.model.trunk,
            output=train_cfg.model.output,
            rescaling=train_cfg.model.rescaling,
            strategy=train_cfg.model.strategy
        )
    )


    print(model(train_transformed['xb'], train_transformed['xt']).shape)
    # training_loop = TrainingLoop(
    #     model=model,
    #     training_params=training_params,
    # )

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()

    end_time = time.time()
    training_time = end_time - start_time

    logger.info(msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")
    
    # ---------------------------- Output folder --------------------------------

    logger.info(msg=f"\nExperiment will be saved at:\n{path_cfg.outputs_path}\n")

