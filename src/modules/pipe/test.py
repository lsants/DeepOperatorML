from __future__ import annotations
import logging
from typing import Any
from .pipeline_config import DataConfig, TestConfig
from . import inference as inf
from ..data_processing import data_loader as dtl
from ..data_processing.postprocessing_helper import run_post_processing

logger = logging.getLogger(__name__)


def test_model(test_cfg_base: TestConfig, exp_cfg_dict: dict[str, Any], data_cfg: DataConfig) -> None:
    checkpoint = dtl.get_trained_model_params(path=test_cfg_base.output_path / exp_cfg_dict['problem'] /
                                              test_cfg_base.experiment_version / 'checkpoints' / 'experiment.pt')

    if exp_cfg_dict['model']['strategy']['name'] == 'two_step':
        exp_cfg_dict['model']['trunk'].update({'inner_config': checkpoint['strategy']['inner_config'],
                                               'T_matrix': checkpoint['model']['trunk.T']}
                                              )

    test_cfg_full = test_cfg_base.with_experiment_data(
        exp_cfg_dict=exp_cfg_dict)
    test_cfg_full = test_cfg_full.with_checkpoint(checkpoint=checkpoint)
    if test_cfg_full.model is None:
        raise TypeError(
            "Model config should not be none after full initialization of test_cfg.")

    # -------------------- Load params and initialize model ---------------------
    inf.inference(test_cfg=test_cfg_full, data_cfg=data_cfg)

    # ------------------------ Process output data ------------------------------
    processed_data_outputs = run_post_processing(
        data_outputs, model, model_to_test_config)

    # run_plotting(processed_data_outputs, test_config, model_to_test_config["IMAGES_FOLDER"])
