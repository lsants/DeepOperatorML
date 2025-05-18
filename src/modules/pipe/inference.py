from __future__ import annotations
import os
import time
import torch
import numpy as np
import logging
from ...modules.pipe.saving import Saver
from ..data_processing.transforms import ToTensor
from ..model.factories.model_factory import ModelFactory
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..model.training_strategies import TwoStepTrainingStrategy, PODTrainingStrategy
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..model import DeepONet

logger = logging.getLogger(__name__)

class TestEvaluator:
    def __init__(self, model: 'DeepONet', error_norm: int | float) -> None:
        self.model = model
        self.error_norm = error_norm

    def __call__(self, g_u: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            test_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return test_error.detach().cpu().numpy()

def inference(trained_model_config: dict[str, Any],
            device: str,
            precision: torch.dtype) -> tuple['DeepONet', dict]:

    # ------------------- Initialize model --------------------------
    model = ModelFactory.initialize_model(trained_model_config, device)
    
    # ------------------- Load data --------------------------
    to_tensor_transform = ToTensor(
        dtype=precision, 
        device=device
    )
    output_keys = trained_model_config["OUTPUT_KEYS"]
    
    processed_data = np.load(trained_model_config['PROCESSED_DATA_PATH'])
    dataset = DeepONetDataset(
        processed_data, 
        transform=to_tensor_transform, 
        output_keys=output_keys
    )

    # ------------------- Prepare inference batch --------------------------
    inference_indices = trained_model_config['TEST_INDICES']
    inference_subset = dataset[inference_indices]
    
    batch = {
        'xb': inference_subset['xb'],
        'xt': dataset.get_trunk(),
        **{key: inference_subset[key] for key in output_keys}
    }

    processed_batch, input_scalers, output_scalers = bt.prepare_batch(
        batch=batch,
        training_params=trained_model_config,
        return_scalers=True
    )

    xb = processed_batch['xb']
    xt = processed_batch['xt']

    # ------------------- Run inference --------------------------
    logger.info("\n\n----------------- Starting inference... --------------\n\n")
    start_time = time.time()
    
    with torch.no_grad():
        if isinstance(model.training_strategy, TwoStepTrainingStrategy):
            raw_preds = model(xb=xb, xt=xt)
        elif isinstance(model.training_strategy, PODTrainingStrategy):
            raw_preds = model(xb=xb)
        else:
            raw_preds = model(xb=xb, xt=xt)
    
    inference_time = time.time() - start_time
    
    # ------------------- Process model outputs --------------------------
    if isinstance(raw_preds, tuple):
        if len(output_keys) == 1:
            preds_norm = {output_keys[0]: raw_preds[0]}
        else:
            preds_norm = {k: v for k, v in zip(output_keys, raw_preds)}
    else:
        preds_norm = raw_preds
    
    # ------------------- Denormalize predictions --------------------------
    denorm_preds = {}
    output_norm = trained_model_config['OUTPUT_NORMALIZATION']
    
    for key in output_keys:
        if output_norm != 'none':
            if output_norm.startswith('minmax'):
                denorm_preds[key] = output_scalers[key].denormalize(preds_norm[key])
            elif output_norm == 'standard':
                denorm_preds[key] = output_scalers[key].destandardize(preds_norm[key])
        else:
            denorm_preds[key] = preds_norm[key]
    
    # ------------------- Calculate metrics --------------------------
    evaluator = TestEvaluator(model, trained_model_config['ERROR_NORM'])
    metrics = {'INFERENCE_TIME': inference_time}
    
    gt_norm = {k: processed_batch[k] for k in output_keys}
    
    errors_norm = {}
    for key in output_keys:
        if output_norm != 'none':
            errors_norm[key] = evaluator(gt_norm[key], preds_norm[key])
    
    errors_phys = {}
    for key in output_keys:
        errors_phys[key] = evaluator(
            batch[key].to(device=device), 
            denorm_preds[key]
        )
        logger.info(f"Test error ({key}) - Physical: {errors_phys[key]:.2%}\n\n")
        if output_norm != 'none':
            logger.info(f"Test error ({key}) - Normalized: {errors_norm[key]:.2%}\n\n")
    
    if errors_norm:
        metrics['ERRORS_NORMALIZED'] = errors_norm
    metrics['ERRORS_PHYSICAL'] = errors_phys
    
    # ------------------- Save results --------------------------
    test_outputs = {
        'predictions': denorm_preds,
        'ground_truth': {k: batch[k].to(device) for k in output_keys},
        'inputs': {
            'xb': xb,
            'xt': xt
        },
        'scalers': {
            'input': input_scalers,
            'output': output_scalers
        }
    }
    
    Saver().save_metrics(
        file_path=os.path.join(trained_model_config["METRICS_PATH"], 'test_metrics.yaml'),
        metrics=metrics
    )
    
    return model, test_outputs