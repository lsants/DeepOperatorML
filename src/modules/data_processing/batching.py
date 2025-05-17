from __future__ import annotations
import torch
from .scaling import Scaling
from .deeponet_dataset import DeepONetDataset
from .transforms import trunk_feature_expansion

def prepare_batch(batch: dict, training_params: dict, 
                input_scalers: dict | None = None,
                output_scalers: dict | None = None,
                return_scalers: bool = False) ->  dict | tuple[dict, dict, dict]:
    """
    Prepares the batch data, including normalization and feature expansion.

    Args:
        batch (dict): The batch data.
        training_params (dict): Dictionary with the normalization parameters.

    Returns:
        dict: The processed batch data.
        scalers: for easier denormalization during inference (optional).
    """
    processed_batch = {}
    device = training_params['DEVICE']
    dtype = getattr(torch, training_params['PRECISION'])
    
    if input_scalers is None:
        input_scalers = {}
        for key in ['xb', 'xt']:
            params = training_params['NORMALIZATION_PARAMETERS'][key]
            if training_params['INPUT_NORMALIZATION'].startswith('minmax'):
                input_scalers[key] = Scaling(min_val=params['min'], max_val=params['max'])
            elif training_params['INPUT_NORMALIZATION'] == 'standard':
                input_scalers[key] = Scaling(mean=params['mean'], std=params['std'])
    
    if output_scalers is None:
        output_scalers = {}
        for key in training_params['OUTPUT_KEYS']:
            params = training_params['NORMALIZATION_PARAMETERS'][key]
            if training_params['OUTPUT_NORMALIZATION'].startswith('minmax'):
                output_scalers[key] = Scaling(min_val=params['min'], max_val=params['max'])
            elif training_params['OUTPUT_NORMALIZATION'] == 'standard':
                output_scalers[key] = Scaling(mean=params['mean'], std=params['std'])
    
    input_norm = training_params['INPUT_NORMALIZATION']
    for key in ['xb', 'xt']:
        if input_norm == 'none':
            processed_batch[key] = batch[key].to(device=device, dtype=dtype)
        else:
            if input_norm.startswith('minmax'):
                target_min = 0.0 if '0_1' in input_norm else -1.0
                target_max = 1.0 if '0_1' in input_norm else 1.0
                processed = input_scalers[key].normalize(batch[key], target_min, target_max)
            elif input_norm == 'standard':
                processed = input_scalers[key].standardize(batch[key])
            processed_batch[key] = processed.to(dtype=dtype, device=device)
    
    output_norm = training_params['OUTPUT_NORMALIZATION']
    for key in training_params['OUTPUT_KEYS']:
        if output_norm == 'none':
            processed_batch[key] = batch[key].to(device=device, dtype=dtype)
        else:
            if output_norm.startswith('minmax'):
                target_min = 0.0 if '0_1' in output_norm else -1.0
                target_max = 1.0 if '0_1' in output_norm else 1.0
                processed = output_scalers[key].normalize(batch[key], target_min, target_max)
            elif output_norm == 'standard':
                processed = output_scalers[key].standardize(batch[key])
            processed_batch[key] = processed.to(dtype=dtype, device=device)
    
    if training_params['TRUNK_FEATURE_EXPANSION'] > 0:
        processed_batch['xt'] = trunk_feature_expansion(
            xt=processed_batch['xt'], 
            n_exp_features=training_params['TRUNK_FEATURE_EXPANSION']
        )
    
    if return_scalers:
        return processed_batch, input_scalers, output_scalers
    return processed_batch

def get_single_batch(dataset: DeepONetDataset, indices, training_params) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, training_params['PRECISION'])
    device = training_params['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = dataset.get_trunk()
    for key in training_params['OUTPUT_KEYS']:
        batch[key] = torch.stack([dataset[idx][key] for idx in indices], dim=0).to(dtype=dtype, device=device)
    return batch