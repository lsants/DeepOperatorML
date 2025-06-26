from __future__ import annotations
import sys
import time
import logging
import dataclasses
from pathlib import Path
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

    transform_pipeline = DeepONetTransformPipeline(
        config=test_cfg.transforms)  # type: ignore

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

    error = error_evaluator(y_truth - y_pred) / error_evaluator(y_truth)

    errors['Physical Error'] = error

    logger.info(
        f"Test error: {error:.3%}, computed in {duration*1000:.3f} ms.")

    saver = Saver()

    saver.save_errors(
        file_path=test_cfg.output_path / test_cfg.problem /
        test_cfg.experiment_version / 'metrics' / 'errors.csv',
        errors=errors
    )
