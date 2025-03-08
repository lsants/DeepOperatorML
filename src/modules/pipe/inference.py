import time
import torch
import numpy
import logging
from ..data_processing import transforms
from ..data_processing.scaling import Scaling
from ..data_processing import data_loader as dtl
from ..data_processing.transforms import ToTensor, Rescale
from ..deeponet.factories.model_factory import ModelFactory
from ..data_processing.deeponet_dataset import DeepONetDataset
from ..deeponet.training_strategies import TwoStepTrainingStrategy, PODTrainingStrategy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..deeponet import DeepONet

logger = logging.getLogger(__name__)

class TestEvaluator:
    def __init__(self, model: 'DeepONet', error_norm: int | float) -> None:
        self.model = model
        self.error_norm = error_norm

    def __call__(self, g_u: torch.Tensor, pred: torch.Tensor) -> numpy.ndarray:
        self.model.eval()
        with torch.no_grad():
            test_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return test_error.detach().cpu().numpy()

def inference(params: dict[str, any]) -> tuple['DeepONet', 
                                               dict[str, torch.Tensor], 
                                               dict[str, torch.Tensor], 
                                               torch.Tensor, 
                                               torch.Tensor, 
                                               dict[str, any]]:
    path_to_data = params['DATAFILE']
    precision = params['PRECISION']
    device = params['DEVICE']
    model_name = params['MODELNAME']
    model_folder = params['MODEL_FOLDER']
    model_location = model_folder + f"model_state_{model_name}.pth"
    
    logger.info(f"\nModel name: \n\n{model_name}\n\n")
    logger.info(f"\nModel loaded from: \n\n{model_location}\n\n")
    logger.info(f"\nData loaded from: \n\n{path_to_data}\n\n")

    model, config_model = ModelFactory.initialize_model(params['MODEL_FOLDER'], 
                                                        params['MODELNAME'], 
                                                        params['DEVICE'], 
                                                        params['PRECISION'])
    
    if config_model['TRAINING_STRATEGY']:
        model.training_phase = 'both'

    evaluator = TestEvaluator(model, config_model['ERROR_NORM'])
    
    to_tensor_transform = ToTensor(dtype=getattr(torch, precision), device=device)
    output_keys = config_model["OUTPUT_KEYS"]

    processed_data = dtl.preprocess_npz_data(path_to_data, 
                                             config_model["INPUT_FUNCTION_KEYS"], 
                                             config_model["COORDINATE_KEYS"], 
                                             direction=config_model["DIRECTION"] if config_model["PROBLEM"] == 'kelvin' else None)
    dataset = DeepONetDataset(processed_data, transform=to_tensor_transform, output_keys=output_keys)
    
    if params['INFERENCE_ON'] == 'train':
        indices_for_inference = config_model['TRAIN_INDICES']
    elif params['INFERENCE_ON'] == 'val':
        indices_for_inference = config_model['VAL_INDICES']
    elif params['INFERENCE_ON'] == 'test':
        indices_for_inference = config_model['TEST_INDICES']
    else:
        indices_for_inference = config_model['TRAIN_INDICES']
    
    inference_dataset = dataset[indices_for_inference]
    
    xb = inference_dataset['xb']  # shape: (N, d)
    xt = dataset.get_trunk()      # trunk features

    ground_truth = {}
    for key in output_keys:
        ground_truth[key] = inference_dataset[key]
    
    xb_scaler = Scaling(
        min_val=config_model['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val=config_model['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    xt_scaler = Scaling(
        min_val=config_model['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val=config_model['NORMALIZATION_PARAMETERS']['xt']['max']
    )
    output_scalers = {}
    for key in output_keys:
        output_scalers[key] = Scaling(
            min_val=config_model['NORMALIZATION_PARAMETERS'][key]['min'],
            max_val=config_model['NORMALIZATION_PARAMETERS'][key]['max']
        )
    
    if config_model['INPUT_NORMALIZATION']:
        xb = xb_scaler.normalize(xb)
        xt = xt_scaler.normalize(xt)
    if config_model['OUTPUT_NORMALIZATION']:
        ground_truth_norm = {key: output_scalers[key].normalize(ground_truth[key]) for key in output_keys}
    
    if config_model['TRUNK_FEATURE_EXPANSION']:
        xt = transforms.trunk_feature_expansion(xt, config_model['TRUNK_FEATURE_EXPANSION'])
    
    logger.info("\n\n----------------- Starting inference... --------------\n\n")
    start_time = time.time()
    if isinstance(model.training_strategy, TwoStepTrainingStrategy):
        if params["PHASE"] == 'trunk':
            preds = model(xt=xt)
        elif params["PHASE"] == 'branch':
            coefs, preds = model(xb=xb)
            preds = coefs
        else:
            preds = model(xb=xb, xt=xt)
    elif isinstance(model.training_strategy, PODTrainingStrategy):
        preds = model(xb=xb)
    else:
        preds = model(xb=xb, xt=xt)
        
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    if len(output_keys) == 1 and not isinstance(preds, dict):
        preds = {output_keys[0]: preds[0]}
    elif len(output_keys) == 2 and not isinstance(preds, dict):
        preds = {k:v for k, v in zip(output_keys, preds)}
    
    if config_model['OUTPUT_NORMALIZATION']:
        preds_norm = {}
        for key in output_keys:
            preds_norm[key], preds[key] = preds[key], output_scalers[key].denormalize(preds[key])
        errors_norm = {}
        for key in output_keys:
            errors_norm[key] = evaluator(ground_truth_norm[key], preds_norm[key])
        config_model['ERRORS_NORMED'] = errors_norm
    else:
        errors_norm = {}
    
    errors = {}

    for key in output_keys:
        errors[key] = evaluator(ground_truth[key], preds[key])
        logger.info(f"Test error for {key} (physical): {errors[key]:.2%}")
        if config_model['OUTPUT_NORMALIZATION']:
            logger.info(f"Test error for {key} (normalized): {errors_norm[key]:.2%}")
    
    config_model['ERRORS_PHYSICAL'] = errors
    config_model['INFERENCE_TIME'] = inference_time
    
    return model, preds, ground_truth, xt, xb, config_model
