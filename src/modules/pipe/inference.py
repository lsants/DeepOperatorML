from __future__ import annotations
import time
import numpy
import logging
from .saving import Saver
from ..model.model_factory import ModelFactory
from ..data_processing import data_loader as dtl
from ..utilities.metrics.errors import ERROR_METRICS
from .pipeline_config import DataConfig, TestConfig
from ..data_processing.deeponet_transformer import DeepONetTransformPipeline

logger = logging.getLogger(__name__)


def inference(test_cfg: TestConfig, data_cfg: DataConfig):

    error_evaluator = ERROR_METRICS[test_cfg.metric]  # type: ignore
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

    predictions = y_pred.detach().numpy()

    duration = time.perf_counter() - start

    errors = {}
    times = {}

    error = (error_evaluator(y_truth - y_pred) /
             error_evaluator(y_truth)).detach().numpy()

    errors['Physical Error'] = {}
    times['inference_time'] = duration

    for i, _ in enumerate(error):
        errors['Physical Error'][data_cfg.targets_labels[i]] = error[i]

    msg = '\n'.join(list(map(lambda x, y: f"{x}: {y:.3%}",
                             data_cfg.targets_labels, error)))
    logger.info(
        f"Test error: \n{msg}, computed in {duration*1000:.3f} ms.")

    data_to_plot = {**{i: j.detach().numpy() for i, j in test_transformed.items()},
                    'predictions': predictions,
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
