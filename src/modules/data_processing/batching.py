import torch
from .scaling import Scaling
from .deeponet_dataset import DeepONetDataset
from .transforms import trunk_feature_expansion


def prepare_batch(batch: dict[str, torch.Tensor], params: dict[str, any]) -> dict[str, torch.Tensor]:
    """
    Prepares the batch data, including normalization and feature expansion.

    Args:
        batch (dict): The batch data.

    Returns:
        dict: The processed batch data.
    """
    processed_batch = {}
    dtype = getattr(torch, params['PRECISION'])
    device = params['DEVICE']

    xb_scaler = Scaling(
        min_val = params['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val = params['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    xt_scaler = Scaling(
        min_val = params['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val = params['NORMALIZATION_PARAMETERS']['xt']['max']
    )

    if params['INPUT_NORMALIZATION']:
        processed_batch['xb'] = xb_scaler.normalize(batch['xb']).to(dtype=dtype, device=device)
        processed_batch['xt'] = xt_scaler.normalize(batch['xt']).to(dtype=dtype, device=device)
    else:
        processed_batch['xb'] = batch['xb'].to(dtype=dtype, device=device)
        processed_batch['xt'] = batch['xt'].to(dtype=dtype, device=device)

    for key in params['OUTPUT_KEYS']:
        scaler = Scaling(
            min_val = params['NORMALIZATION_PARAMETERS'][key]['min'],
            max_val = params['NORMALIZATION_PARAMETERS'][key]['max']
        )
        if params['OUTPUT_NORMALIZATION']:
            processed_batch[key] = scaler.normalize(batch[key]).to(dtype=dtype, device=device)
        else:
            processed_batch[key] = batch[key].to(dtype=dtype, device=device)

    if params['TRUNK_FEATURE_EXPANSION'] > 0:
        processed_batch['xt'] = trunk_feature_expansion(
            processed_batch['xt'], params['TRUNK_FEATURE_EXPANSION']
        )

    return processed_batch

def get_single_batch(dataset: DeepONetDataset, indices, params) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, params['PRECISION'])
    device = params['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = dataset.get_trunk()
    for key in params['OUTPUT_KEYS']:
        batch[key] = torch.stack([dataset[idx][key] for idx in indices], dim=0).to(dtype=dtype, device=device)
    return batch