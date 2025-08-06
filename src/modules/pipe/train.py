import dataclasses
import logging
import time
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from src.modules.pipe.saving import Saver
from src.modules.pipe.plot_training import plot_training
from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
from src.modules.pipe.deeponet_training_loop import DeepONetTrainingLoop
from src.modules.models.deeponet.deeponet_factory import DeepONetFactory
from src.modules.models.deeponet.dataset import preprocessing_utils as dtl
from src.modules.models.deeponet.dataset.deeponet_sampler import DeepONetSampler
from src.modules.models.deeponet.dataset.deeponet_dataset import DeepONetDataset
from src.modules.models.deeponet.dataset.deeponet_transform import DeepONetTransformPipeline
from src.modules.models.deeponet.config import TrainConfig, DataConfig, ExperimentConfig, PathConfig

logger = logging.getLogger(name=__name__)

def train_model(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
):

    torch.random.manual_seed(train_cfg.seed)
    path_cfg = PathConfig.from_data_config(data_cfg=data_cfg) # TODO: only create paths after model finishes training.

    exp_cfg = ExperimentConfig.from_dataclasses(
        data_cfg=data_cfg, 
        train_cfg=train_cfg, 
        path_cfg=path_cfg
    )

    stats = dtl.get_stats(
        data=data_cfg.scalers,
        keys=data_cfg.features + data_cfg.targets
    )

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

    train_data, val_data, _ = dtl.get_split_data(
        data=data_cfg.data,
        split_indices=data_cfg.split_indices,
        features_keys=data_cfg.features,
        targets_keys=data_cfg.targets
    )
    train_transformed = dtl.get_transformed_data(
        data=train_data,
        features_keys=data_cfg.features,
        targets_keys=data_cfg.targets,
        transform_pipeline=transform_pipeline
    )
    val_transformed = dtl.get_transformed_data(
        data=val_data,
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

    train_sampler = DeepONetSampler(
        num_branch_samples=train_branch_samples,
        branch_batch_size=train_cfg.branch_batch_size,
        num_trunk_samples=train_trunk_samples,
        trunk_batch_size=train_cfg.trunk_batch_size,
        shuffle=False
    )

    val_sampler = DeepONetSampler(
        num_branch_samples=val_branch_samples,
        branch_batch_size=train_cfg.branch_batch_size,
        num_trunk_samples=val_trunk_samples,
        trunk_batch_size=train_cfg.trunk_batch_size,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=lambda x: x[0]
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        collate_fn=lambda x: x[0]
    )

    logger.info(
        msg=f"\nxb: {train_dataset[:]['xb'].shape},\nxt: {train_dataset[:]['xt'].shape},\ng_u: {train_dataset[:]['g_u'].shape}")
    logger.info(
        msg=f"Training branch samples: {train_branch_samples} samples, train branch batch size: {train_cfg.branch_batch_size}\nTraining trunk samples: {train_trunk_samples} samples, train trunk batch size: {train_cfg.trunk_batch_size}")

    # ------------------------------------ Initialize model & train loop -----------------------------

    model, train_strategy = DeepONetFactory.create_for_training(
        config=DeepONetConfig(
            branch=train_cfg.model.branch,
            trunk=train_cfg.model.trunk,
            bias=train_cfg.model.bias,
            output=train_cfg.model.output,
            rescaling=train_cfg.model.rescaling,
            strategy=train_cfg.model.strategy
        )
    )

    model.to(device=exp_cfg.device, dtype=exp_cfg.precision) # type: ignore

    PathConfig.create_directories(path_cfg)

    saver = Saver()

    saver.save_transform_pipeline(
        file_path=path_cfg.checkpoints_path,
        transform_pipeline=transform_pipeline,
    )

    loop = DeepONetTrainingLoop(
        model=model,
        strategy=train_strategy,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=train_cfg.device,
        checkpoint_dir=path_cfg.checkpoints_path,
        sampler=train_sampler,
        label_map=data_cfg.targets_labels
    )

    # ----------------------------------------- Train loop ---------------------------------
    start_time = time.time()

    loop.run()

    end_time = time.time()

    training_time = end_time - start_time

    times = {'training_time': training_time}

    history = loop.history.get_history()

    fig = plot_training(
        history=history, plot_config=dataclasses.asdict(train_cfg))

    exp_cfg = exp_cfg.get_serializable_config()


    if hasattr(train_strategy, 'final_trunk_config'):
        final_model_config = deepcopy(exp_cfg.model)
        final_model_config.trunk = train_strategy.final_trunk_config  # type: ignore
        final_model_config.trunk.pod_basis = None
        if final_model_config.trunk.inner_config is not None:
            final_model_config.trunk.inner_config.pod_basis = None
        final_model_config.branch = train_strategy.final_branch_config  # type: ignore
    else:
        final_model_config = exp_cfg.model

    exp_dict = dataclasses.asdict(exp_cfg)
    try:
        del exp_dict['strategy']['pod_basis']
        del exp_dict['strategy']['pod_mean']
    except KeyError:
        pass

    saver.save_model_info(
        file_path=path_cfg.outputs_path / 'experiment_config.yaml',
        model_info={
            **exp_dict,
            "model": dataclasses.asdict(final_model_config)
        }
    )

    saver.save_plots(
        file_path=path_cfg.plots_path / 'training_history.png',
        figure=fig
    )

    saver.save_time(
        file_path=path_cfg.metrics_path / 'training_time.yaml',
        times=times
    )

    saver.save_history(
        file_path=path_cfg.metrics_path / 'train_metrics.yaml',
        history=history
    )
    logger.info(msg=f"Experiment saved at {path_cfg.outputs_path}")

    logger.info(
        msg=f"\n----------------------------------------Training concluded in: {training_time:.2f} seconds---------------------------\n")
