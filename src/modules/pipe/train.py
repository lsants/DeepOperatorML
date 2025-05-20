import os
import time
import logging
from src.modules.pipe.pipeline_config import TrainConfig, DataConfig, ExperimentConfig, PathConfig
from .saving import Saver
from .training_loop import TrainingLoop
from ..utilities import dir_functions
from torch.utils.data import DataLoader
from ..data_processing import data_loader as dtl
from ..model.model_factory import ModelFactory
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..data_processing.deeponet_sampler import DeepONetSampler
from ..data_processing.deeponet_transformer import DeepONetTransformPipeline

logger = logging.getLogger(name=__name__)

def train_model(
                data_cfg: DataConfig,
                train_cfg: TrainConfig,
            ):

    data = data_cfg.data
    split_indices = data_cfg.split_indices
    features_keys = data_cfg.features
    branch_key = features_keys[0]
    trunk_key = features_keys[1]
    targets_keys = data_cfg.targets

    experiment_cfg = ExperimentConfig.from_dataclasses(
        data_cfg=data_cfg,
        train_cfg=train_cfg
    )

    path_cfg = PathConfig.from_data_config(data_cfg=data_cfg)
    PathConfig.create_directories(path_cfg)

    train_indices = (split_indices[f'{branch_key}_train'], split_indices[f'{trunk_key}_train'])
    val_indices = (split_indices[f'{branch_key}_val'],        split_indices[f'{trunk_key}_val'])

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
    
    transform_pipeline = DeepONetTransformPipeline(train_cfg.transforms)

    saver = Saver()
    saver.save_transform_pipeline(
        file_path = path_cfg.checkpoints_path,
        transform_pipeline = transform_pipeline,
    )

    train_transformed = {
        features_keys[0]: transform_pipeline.transform_branch(train_data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(train_data[features_keys[1]]),
        **{k: train_data[k] for k in targets_keys}  # Preserve original outputs
    }

    val_transformed = {
        features_keys[0]: transform_pipeline.transform_branch(val_data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(val_data[features_keys[1]]),
        **{k: val_data[k] for k in targets_keys}  # Preserve original outputs
    }

    train_dataset = DeepONetDataset(
        data=train_transformed,
        feature_labels=features_keys,
        output_labels=targets_keys,
    )
    val_dataset = DeepONetDataset(
        data=val_transformed,
        feature_labels=features_keys,
        output_labels=targets_keys,
    )

    train_branch_samples = len(train_dataset[:][features_keys[0]])
    train_trunk_samples = len(train_dataset[:][features_keys[1]])

    train_sampler = DeepONetSampler(num_branch_samples=train_branch_samples,
                              branch_batch_size=train_cfg.branch_batch_size,
                              num_trunk_samples=train_trunk_samples,
                              trunk_batch_size=train_cfg.trunk_batch_size['TRUNK_BATCH_SIZE'], #type: ignore
                              )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=lambda x : x[0]
    )
    # ------------------------------------ Initialize model -----------------------------

    # model = ModelFactory.create(
    #     problem_cfg=problem_cfg,
    #     train_cfg=train_cfg,
    #     input_dim=transformed_train_data['inputs'].shape[-1],
    #     transform_mgr=transform_mgr,
    # )


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

