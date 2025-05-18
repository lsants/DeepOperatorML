from __future__ import annotations
import torch
import logging
import numpy as np
from ...modules.data_processing.scaling import Scaling
from ...modules.data_processing.data_loader import don_to_meshgrid

logger = logging.getLogger(__file__)

def postprocess_for_2D_plot(model, plot_config: dict[str, any], model_config: dict[str, any], 
                            branch_features: torch.Tensor, trunk_features: torch.Tensor,
                            ground_truth: torch.Tensor, preds: torch.Tensor) -> dict[str, np.ndarray]:
    processed_data = {}

    # -------------- Prepare branch data --------------

    xb_keys = model_config["INPUT_FUNCTION_KEYS"]

    xb_scaler = Scaling(
        min_val=model_config['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val=model_config['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    if model_config['INPUT_NORMALIZATION']:
        branch_features = xb_scaler.denormalize(branch_features)
    branch_tuple = don_to_meshgrid(branch_features)
    branch_map = {k:v for k, v in zip(xb_keys, branch_tuple)}
    processed_data["branch_map"] = branch_map
    processed_data["branch_features"] = np.array(branch_features)

    # -------------- Prepare trunk data --------------

    xt_scaler = Scaling(
        min_val=model_config['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val=model_config['NORMALIZATION_PARAMETERS']['xt']['max']
    )
    xt_plot = trunk_features
    if model_config['TRUNK_FEATURE_EXPANSION']:
        xt_plot = xt_plot[:, : xt_plot.shape[-1] // (1 + 2 * model_config['TRUNK_FEATURE_EXPANSION'])]
    if model_config['INPUT_NORMALIZATION']:
        xt_plot = xt_scaler.denormalize(xt_plot)

    if "COORDINATE_KEYS" not in model_config:
        raise ValueError("COORDINATE_KEYS must be provided in the configuration.")
    coordinate_keys = model_config["COORDINATE_KEYS"]
    coords_tuple = don_to_meshgrid(xt_plot)
    if len(coords_tuple) != len(coordinate_keys):
        raise ValueError("Mismatch between number of coordinates in trunk data and COORDINATE_KEYS.")
    
    coordinates_map = {k: v for k, v in zip(coordinate_keys, coords_tuple)}
    coord_index_map = {coord: index for index, coord in enumerate(coordinates_map)}
    coords_2D_index_map = {k: v for k, v in coord_index_map.items() if k in plot_config["AXES_TO_PLOT"]}

    if len(coord_index_map) > 2:
        index_to_remove_coords = [coord_index_map[coord] for coord in coord_index_map if coord not in coords_2D_index_map][0]
    else:
        index_to_remove_coords = None
    col_indices = [index for index in coord_index_map.values() if index != index_to_remove_coords]
    coords_2D_map = {k : v for k, v in coordinates_map.items() if k in plot_config["AXES_TO_PLOT"]}

    processed_data["coords_2D"] = coords_2D_map
    processed_data["trunk_features"] = xt_plot


    if len(coord_index_map) > 2:
        processed_data["trunk_features_2D"] = processed_data["trunk_features"][ : , col_indices]
    else:
        processed_data["trunk_features_2D"] = processed_data["trunk_features"]
    
    # ------------------ Prepare outputs ---------------------

    output_keys = model_config["OUTPUT_KEYS"]
    if len(output_keys) == 2:
        truth_field = ground_truth[output_keys[0]] + ground_truth[output_keys[1]] * 1j
        pred_field = preds[output_keys[0]] + preds[output_keys[1]] * 1j
    else:
        truth_field = ground_truth[output_keys[0]]
        pred_field = preds[output_keys[0]]

    truth_field = process_outputs_to_plot_format(truth_field, coords_tuple)
    pred_field = process_outputs_to_plot_format(pred_field, coords_tuple)

    trunk_output = model.trunk.forward(trunk_features).T 
    branch_output = model.branch.forward(branch_features) 
    basis_modes = process_basis_to_plot_format(trunk_output, coords_tuple, len(output_keys), basis_config=model_config["BASIS_CONFIG"]) # (n_basis, *coords, n_channels)
    coefficients = process_coefficients_to_plot_format(branch_output, len(output_keys), output_handling=model_config['OUTPUT_HANDLING']) # (input_functions, n_basis, n_channels)
    
    if basis_modes.shape[0] > model_config.get('BASIS_FUNCTIONS'):
        split_1 = basis_modes[ : model_config.get('BASIS_FUNCTIONS')]
        split_2 = basis_modes[model_config.get('BASIS_FUNCTIONS') : ]
        basis_modes = np.concatenate([split_1, split_2], axis=-1)

    truth_slicer = [slice(None)] * truth_field.ndim
    pred_slicer = [slice(None)] * pred_field.ndim
    basis_slicer = [slice(None)] * basis_modes.ndim
    if index_to_remove_coords is not None:  # Fix: Check for 'is not None' instead of truthy value
        processed_data["index_to_remove_coords"] = index_to_remove_coords
        # Fix: Remove +2 from truth and pred slicers
        truth_slicer[index_to_remove_coords] = 0
        pred_slicer[index_to_remove_coords] = 0
        basis_slicer[index_to_remove_coords + 1] = 0  # Basis slicer remains correct
    
    basis_modes_sliced = basis_modes[tuple(basis_slicer)]
    
    processed_data["output_keys"] = output_keys
    processed_data["truth_field"] = truth_field
    processed_data["pred_field"] = pred_field
    processed_data["truth_field_2D"] = truth_field[tuple(truth_slicer)]
    processed_data["pred_field_2D"] = pred_field[tuple(pred_slicer)]
    processed_data["basis_functions_2D"] = basis_modes_sliced
    processed_data["coefficients"] = coefficients
    
    logger.info(f"\nOutputs shape: {pred_field.shape}\n")
    logger.info(f"\n2D Outputs shape: {processed_data['pred_field_2D'].shape}\n")
    logger.info(f"\n2D Truths shape: {processed_data['truth_field_2D'].shape}\n")
    logger.info(f"\n2D Basis functions shape: {processed_data['basis_functions_2D'].shape}\n")

    return processed_data

def process_outputs_to_plot_format(output: torch.Tensor, coords: tuple | list | np.ndarray) -> np.ndarray:
    """
    Reshapes the output from DeepONet into a meshgrid format for plotting.
    
    Args:
        output (Tensor or ndarray): The network output, with shape either
            (branch_data.shape[0], trunk_size) or 
            (branch_data.shape[0], trunk_size, n_basis).
        coords (tuple or list or ndarray): The coordinate arrays that were
            used to generate the trunk. If multiple coordinate arrays are provided,
            they should be in a tuple/list (e.g. (x_values, y_values, z_values)).
            If a single array is provided, it is assumed to be 1D.
    
    Returns:
        ndarray: Reshaped output with shape 
            (branch_data.shape[0], n_basis, len(coords[0]), len(coords[1]), ...).
            For a 2D problem with a single basis, for example, the output shape
            will be (N_branch, 1, len(coord1), len(coord2)).
    """
    if isinstance(coords, (list, tuple)):
        grid_shape = tuple(len(c) for c in coords)
    else:
        grid_shape = (len(coords),)
    
    output = output.detach().cpu().numpy()

    if output.ndim == 2:
        N_branch, trunk_size = output.shape
        if np.prod(grid_shape) != trunk_size:
            raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, *grid_shape, 1)
    elif output.ndim == 3:
        N_branch, outputs, trunk_size = output.shape
        if np.prod(grid_shape) != trunk_size:
                raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, *grid_shape, outputs)

    else:
        raise ValueError("Output must be either 2D or 3D.")
    
    return reshaped

def process_basis_to_plot_format(basis: torch.Tensor, coords: tuple | list | np.ndarray, n_channels: int, basis_config: str) -> np.ndarray:
    if isinstance(coords, (list, tuple)):
        grid_shape = tuple(len(c) for c in coords)
    else:
        grid_shape = (len(coords),)

    
    basis = basis.detach().cpu().numpy()
    trunk_output_size, trunk_batch_size = basis.shape
    if np.prod(grid_shape) != trunk_batch_size:
            raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
    intermediate_reshaping = basis.reshape(trunk_output_size, *grid_shape)
    if basis_config == 'multiple':
        modes_pre_concat = [intermediate_reshaping[ i * trunk_output_size // n_channels : (i + 1) * (trunk_output_size // n_channels), ... , None] for i in range(n_channels)]
        basis_reshaped = np.concatenate(modes_pre_concat, axis=-1)
    else:
        modes_pre_concat = [intermediate_reshaping[ i * trunk_output_size : (i + 1) * trunk_output_size, ... , None] for i in range(n_channels)]
        basis_reshaped = np.concatenate(modes_pre_concat, axis=0)

    return basis_reshaped

def process_coefficients_to_plot_format(coefficients: torch.Tensor, n_channels: int, output_handling: str) -> np.ndarray:
    coefficients = coefficients.detach().cpu().numpy()
    _, n_modes = coefficients.shape[0], coefficients.shape[1]
    if output_handling in {'share_trunk', 'split_outputs'}:
        intermediate_reshaping = [coefficients[ : , i * n_modes // n_channels : (i + 1) * (n_modes // n_channels), None] for i in range(n_channels)]
    else:
        intermediate_reshaping = [coefficients[ : , i * n_modes  : (i + 1) * n_modes, None] for i in range(n_channels)]
    coefficients_reshaped = np.concatenate(intermediate_reshaping, axis=-1)
    return coefficients_reshaped