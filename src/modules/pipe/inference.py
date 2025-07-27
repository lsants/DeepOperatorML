from __future__ import annotations
import time
import numpy
import logging
from src.modules.pipe.saving import Saver
from src.modules.model.model_factory import ModelFactory
from src.modules.data_processing import data_loader as dtl
from src.modules.utilities.metrics.errors import ERROR_METRICS
from src.modules.pipe.pipeline_config import DataConfig, TestConfig
from src.modules.data_processing.deeponet_transform import DeepONetTransformPipeline

logger = logging.getLogger(__name__)


def inference(test_cfg: TestConfig, data_cfg: DataConfig):

    metric = ERROR_METRICS[test_cfg.metric]  # type: ignore
    model_params = test_cfg.checkpoint['model']  # type: ignore

    _, _, test_data = dtl.get_split_data(data=data_cfg.data,
                                         split_indices=data_cfg.split_indices,
                                         features_keys=data_cfg.features,
                                         targets_keys=data_cfg.targets
                                         )

    stats = dtl.get_stats(data=data_cfg.scalers,
                          keys=data_cfg.features + data_cfg.targets)

    transform_pipeline = DeepONetTransformPipeline(
        config=test_cfg.transforms)  # type: ignore

    transform_pipeline.set_branch_stats(
        stats=stats[data_cfg.features[0]]
    )
    transform_pipeline.set_trunk_stats(
        stats=stats[data_cfg.features[1]]
    )
    transform_pipeline.set_target_stats(
        stats=stats[data_cfg.targets[0]]
    )

    test_transformed = dtl.get_transformed_data(data=test_data,
                                                features_keys=data_cfg.features,
                                                targets_keys=data_cfg.targets,
                                                transform_pipeline=transform_pipeline
                                                )


    model = ModelFactory.create_for_inference(
        saved_config=test_cfg.model, state_dict=model_params)  # type: ignore

    y_truth = test_transformed[data_cfg.targets[0]]

    start = time.perf_counter()

    y_pred = model(test_transformed[data_cfg.features[0]],
                   test_transformed[data_cfg.features[1]]
                   )

    duration = time.perf_counter() - start

    errors = {}
    times = {}

    abs_error = metric(y_truth - y_pred).detach().numpy()
    norm_truth = metric(y_truth).detach().numpy()


    errors['Physical Error'] = {}
    errors['Normalized Error'] = {}
    times['inference_time'] = duration

    if test_cfg.transforms.target.normalization is not None:
        y_pred = transform_pipeline.inverse_transform(tensor=y_pred, component='target')
    
    for i, _ in enumerate(abs_error):
        if test_cfg.transforms.target.normalization is None:
            errors['Physical Error'][data_cfg.targets_labels[i]] = (abs_error / norm_truth)[i]
        else:
            errors['Normalized Error'][data_cfg.targets_labels[i]] = (abs_error / norm_truth)[i]
            errors['Physical Error'][data_cfg.targets_labels[i]] = (stats['g_u']['std'] * abs_error \
                / (stats['g_u']['std'] * norm_truth + stats['g_u']['mean']))[i]
            
    msg = '\n'.join(list(map(lambda x, y: f"{x}: {y:.3%}",
                             data_cfg.targets_labels, errors['Physical Error'].values())))
    
    
    logger.info(
        f"Test error: \n{msg}, computed in {duration*1000:.3f} ms.")

    data_to_plot = {**{i: j for i, j in data_cfg.data.items()},
                    'predictions': y_pred.detach().numpy(),
                    'branch_output': model.branch(test_transformed[data_cfg.features[0]]).detach().numpy(),
                    'trunk_output': model.trunk(test_transformed[data_cfg.features[1]]).detach().numpy(),
                    'bias': model.bias.bias.detach().numpy()
                    }

    saver = Saver()

    saver.save_errors(
        file_path=test_cfg.output_path / test_cfg.problem /  # type: ignore
        test_cfg.experiment_version / 'metrics' / 'test_metrics.yaml',
        errors=errors
    )
    saver.save_time(
        file_path=test_cfg.output_path / test_cfg.problem /  # type: ignore
        test_cfg.experiment_version / 'metrics' / 'test_time.yaml',
        times=times
    )

    numpy.savez(test_cfg.output_path / test_cfg.problem /  # type: ignore
                test_cfg.experiment_version / 'aux' / 'output_data.npz', **data_to_plot)

    logger.info(
        # type: ignore
        f"Saved to {test_cfg.output_path / test_cfg.problem / test_cfg.experiment_version}"
    )
