from __future__ import annotations
import numpy as np
from typing import Any
from collections.abc import Mapping
from src.modules.data_processing.deeponet_transform import DeepONetTransformPipeline

def slice_data(data: Mapping[str, np.ndarray],
               feature_keys: list[str],
               target_keys: list[str],
               split_indices: tuple[np.ndarray, ...],
               trunk_slice=False) -> dict[str, np.ndarray]:
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
    transformed_data = {
        features_keys[0]: transform_pipeline.transform_branch(data[features_keys[0]]),
        features_keys[1]: transform_pipeline.transform_trunk(data[features_keys[1]]),
        # Preserve original outputs
        **{k: transform_pipeline.transform_target(data[k]) for k in targets_keys}
    }

    return transformed_data


def get_stats(data: dict[str, np.ndarray], keys: list[str]) -> dict[str, Any]:
    """Compute statistics for normalization."""
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

