import time
import torch
import logging
from modules.utilities import dir_functions
from modules.data_processing import preprocessing as ppr
from modules.pipe.model_factory import initialize_model
from modules.data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)

class TestEvaluator:
    def __init__(self, model, error_norm):
        self.model = model
        self.error_norm = error_norm

    def __call__(self, g_u, pred):
        self.model.eval()
        with torch.no_grad():
            test_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return test_error.detach().cpu().numpy()

def inference(p: dict):
    # Load configuration from YAML.
    path_to_data = p['DATAFILE']
    precision = p['PRECISION']
    device = p['DEVICE']
    model_name = p['MODELNAME']
    model_folder = p['MODEL_FOLDER']
    model_location = model_folder + f"model_state_{model_name}.pth"
    
    logger.info(f"\nModel name: \n\n{model_name}\n\n")
    logger.info(f"\nModel loaded from: \n\n{model_location}\n\n")
    logger.info(f"\nData loaded from: \n\n{path_to_data}\n\n")

    model, config_model = initialize_model(p['MODEL_FOLDER'], p['MODELNAME'], p['DEVICE'], p['PRECISION'])
    
    if config_model['TRAINING_STRATEGY']:
        model.training_phase = 'both'

    evaluator = TestEvaluator(model, config_model['ERROR_NORM'])
    
    to_tensor_transform = ppr.ToTensor(dtype=getattr(torch, precision), device=device)
    output_keys = config_model["OUTPUT_KEYS"]

    processed_data = ppr.preprocess_npz_data(path_to_data, 
                                             config_model["INPUT_FUNCTION_KEYS"], 
                                             config_model["COORDINATE_KEYS"], 
                                             direction=config_model["DIRECTION"] if config_model["PROBLEM"] == 'kelvin' else None)
    dataset = DeepONetDataset(processed_data, transform=to_tensor_transform, output_keys=output_keys)
    
    if p['INFERENCE_ON'] == 'train':
        indices_for_inference = config_model['TRAIN_INDICES']
    elif p['INFERENCE_ON'] == 'val':
        indices_for_inference = config_model['VAL_INDICES']
    elif p['INFERENCE_ON'] == 'test':
        indices_for_inference = config_model['TEST_INDICES']
    else:
        indices_for_inference = config_model['TRAIN_INDICES']
    
    inference_dataset = dataset[indices_for_inference]
    
    # Get branch and trunk inputs.
    xb = inference_dataset['xb']  # shape: (N, d)
    xt = dataset.get_trunk()      # trunk features

    # Get ground truth outputs.
    ground_truth = {}
    for key in output_keys:
        ground_truth[key] = inference_dataset[key]
    
    # Initialize normalization functions.
    xb_scaler = ppr.Scaling(
        min_val=config_model['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val=config_model['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    xt_scaler = ppr.Scaling(
        min_val=config_model['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val=config_model['NORMALIZATION_PARAMETERS']['xt']['max']
    )
    output_scalers = {}
    for key in output_keys:
        output_scalers[key] = ppr.Scaling(
            min_val=config_model['NORMALIZATION_PARAMETERS'][key]['min'],
            max_val=config_model['NORMALIZATION_PARAMETERS'][key]['max']
        )
    
    if config_model['INPUT_NORMALIZATION']:
        xb = xb_scaler.normalize(xb)
        xt = xt_scaler.normalize(xt)
    if config_model['OUTPUT_NORMALIZATION']:
        ground_truth_norm = {key: output_scalers[key].normalize(ground_truth[key]) for key in output_keys}
    
    if config_model['TRUNK_FEATURE_EXPANSION']:
        xt = ppr.trunk_feature_expansion(xt, config_model['TRUNK_EXPANSION_FEATURES_NUMBER'])
    
    # Evaluation.
    logger.info("\n\n----------------- Starting inference... --------------\n\n")
    start_time = time.time()
    if config_model['TRAINING_STRATEGY'].lower() == 'two_step':
        if p["PHASE"] == 'trunk':
            preds = model(xt=xt)
        elif p["PHASE"] == 'branch':
            coefs, preds = model(xb=xb)
            preds = coefs
        else:
            preds = model(xb=xb, xt=xt)
    elif config_model['TRAINING_STRATEGY'] == 'pod':
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
            preds_norm[key] = output_scalers[key].normalize(preds[key])
        errors_norm = {}
        for key in output_keys:
            errors_norm[key] = evaluator(ground_truth_norm[key], preds_norm[key])
        for key in output_keys:
            preds[key] = output_scalers[key].denormalize(preds_norm[key])
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
