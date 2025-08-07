from __future__ import annotations
import numpy as np
from typing import Any
from collections.abc import Mapping
from src.modules.models.deeponet.dataset.deeponet_transform import DeepONetTransformPipeline

def slice_data(data: Mapping[str, np.ndarray],
               feature_keys: list[str],
               target_keys: list[str],
               split_indices: tuple[np.ndarray, ...],
               trunk_slice=False) -> dict[str, np.ndarray]:
    """
    Slices a dictionary of data based on provided branch and trunk indices.

    This utility function is used to extract a subset of data for a specific
    split (e.g., training, validation) from a larger dataset. It handles the
    slicing of branch, trunk, and target data.

    Args:
        data (Mapping[str, np.ndarray]): The full data dictionary containing
                                         all feature and target data.
        feature_keys (List[str]): A list of keys for the branch and trunk data.
                                  The first key is for the branch, the second for the trunk.
        target_keys (List[str]): A list of keys for the target data.
        split_indices (Tuple[np.ndarray, ...]): A tuple containing two arrays
                                                of indices: one for the branch data and
                                                one for the trunk data.
        trunk_slice (bool, optional): If True, the trunk data is also sliced.
                                      Otherwise, the full trunk data is returned.
                                      Defaults to False.

    Returns:
        Dict[str, np.ndarray]: A new dictionary containing the sliced data.
    """
    branch_label = feature_keys[0]
    branch_splits = split_indices[0]
    trunk_label = feature_keys[1]
    trunk_splits = split_indices[1]
    target_labels = target_keys
    branch_data = data[branch_label]
    trunk_data = data[trunk_label]
    target_data = {key: data[key] for key in target_labels}

    split_branch_data = branch_data[branch_splits]
    split_trunk_data = trunk_data
    split_target_data = {key: val[branch_splits]
                         for key, val in target_data.items()}

    if trunk_slice:
        split_trunk_data = trunk_data[trunk_splits]
        split_target_data = {key: val[branch_splits][:, trunk_splits]
                             for key, val in target_data.items()}
    split_data = {
        branch_label: split_branch_data,
        trunk_label: split_trunk_data,
        **split_target_data
    }
    return split_data


def get_split_data(data: Any,
                   split_indices: dict[str, np.ndarray],
                   features_keys: list[str],
                   targets_keys: list[str]) -> tuple[dict[str, np.ndarray[Any, Any]], ...]:
    """
    Splits a dataset into training, validation, and testing sets.

    This function uses pre-defined indices for each split to create three
    separate data dictionaries, which can then be used for model training,
    validation, and evaluation.

    Args:
        data (Any): The full data dictionary.
        split_indices (Dict[str, np.ndarray]): A dictionary containing the
                                               indices for each split, with keys
                                               like 'branch_key_train', 'trunk_key_val', etc.
        features_keys (List[str]): The keys for the branch and trunk data.
        targets_keys (List[str]): The keys for the target data.

    Returns:
        Tuple[Dict[str, np.ndarray], ...]: A tuple containing the train,
                                          validation, and test data dictionaries.
    """

    branch_key = features_keys[0]
    trunk_key = features_keys[1]

    train_indices = (split_indices[f'{branch_key}_train'],
                     split_indices[f'{trunk_key}_train'])
    val_indices = (split_indices[f'{branch_key}_val'],
                   split_indices[f'{trunk_key}_train'])
    test_indices = (split_indices[f'{branch_key}_test'],
                    split_indices[f'{trunk_key}_test'])

    train_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=train_indices
    )

    val_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=val_indices
    )

    test_data = slice_data(
        data=data,
        feature_keys=features_keys,
        target_keys=targets_keys,
        split_indices=test_indices
    )

    return train_data, val_data, test_data


def get_transformed_data(data: Any, features_keys: list[str], targets_keys: list[str], transform_pipeline: DeepONetTransformPipeline) -> dict[str, Any]:
    """
    Applies a DeepONetTransformPipeline to a dictionary of data.

    This function iterates through the feature and target keys, applying the
    appropriate transformation methods from the provided pipeline.

    Args:
        data (Any): The dictionary of data to transform.
        features_keys (List[str]): The keys for the branch and trunk data.
        targets_keys (List[str]): The keys for the target data.
        transform_pipeline (DeepONetTransformPipeline): The pipeline object
                                                       containing the transformation
                                                       logic and statistics.

    Returns:
        Dict[str, Any]: A new dictionary with the transformed data as tensors.
    """
    transformed_data = {
        features_keys[0]: transform_pipeline.transform_branch(data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(data[features_keys[1]]),
        # Preserve original outputs
        **{k: transform_pipeline.transform_target(data[k]) for k in targets_keys}
    }

    return transformed_data

def get_stats(data: dict[str, np.ndarray], keys: list[str]) -> dict[str, Any]:
    """
    Retrieves pre-computed statistics from a data dictionary.

    This utility function is useful for loading normalization statistics that
    have been saved separately. It expects the statistics to be stored with
    suffixes like '_mean', '_std', '_min', and '_max'.

    Args:
        data (Dict[str, np.ndarray]): The dictionary containing the statistics.
        keys (List[str]): The base keys for which to retrieve statistics.

    Returns:
        Dict[str, Any]: A nested dictionary with the retrieved statistics.

    Raises:
        KeyError: If a required key (e.g., 'key_mean') is not found in the data.
    """
    stats = {}
    for key in keys:
        check = any(key in k for k in data.keys())
        if not check:
            raise KeyError(f"Key {key} not found in data.")
        else:
            stats[key] = {
                'mean': data[f'{key}_mean'],
                'std': data[f'{key}_std'],
                'min': data[f'{key}_min'],
                'max': data[f'{key}_max'],
            }
    return stats

