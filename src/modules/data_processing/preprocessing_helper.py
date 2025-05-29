from __future__ import annotations
import hashlib
import numpy as np
from torch import values_copy
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any
from .deeponet_dataset import DeepONetDataset

def validate_data_structure(data: dict, config: dict) -> dict:
    """Validate feature/target dimensions based on config relationships."""
    features = config['data_labels']['features']
    targets = config['data_labels']['targets']
    
    for name in features + targets:
        if name not in data:
            raise ValueError(f"Missing array '{name}' in processed data")
    
    for target in targets:
        target_shape = data[target].shape
        if len(target_shape) != 2:
            raise ValueError(f"Target '{target}' must be 2-dimensional")
            
        if target_shape[0] != data[features[0]].shape[0]:
            raise ValueError(f"Target '{target}' rows don't match '{features[0]}' samples")
            
        if target_shape[1] != data[features[1]].shape[0]:
            raise ValueError(f"Target '{target}' columns don't match '{features[1]}' samples")
    
    return {
        'features': {f: data[f].shape[0] for f in features},
        'targets': {t: data[t].shape for t in targets}
    }

def get_data_shapes(data: dict[str, Any], config: dict[str, Any]) -> dict[str, tuple[int]]:
    """Get dataset shapes."""
    features = config['data_labels']['features']
    targets = config['data_labels']['targets']
    data_shapes = {f: data[f].shape for f in features} | {t: data[t].shape for t in targets}
    return data_shapes

def get_sample_sizes(data: dict, config: dict) -> dict[str, int]:
    """Get sample sizes for all features based on config relationships."""
    dim_info = validate_data_structure(data, config)
    return dim_info['features']

def split_features(
    sample_sizes: dict[str, int], 
    split_ratios: list[float],
    seed: int
) -> dict[str, dict[str, np.ndarray]]:
    """Generate splits for multiple features with independent sample sizes.
    
    Returns:
        {
            'xb': {'train': ..., 'val': ..., 'test': ...},
            'xt': {'train': ..., 'val': ..., 'test': ...},
            ...
        }
    """
    rng = np.random.default_rng(seed)
    splits = {}
    
    for feature, num_samples in sample_sizes.items():
        indices = rng.permutation(num_samples)
        
        train_end = int(split_ratios[0] * num_samples)
        val_end = train_end + int(split_ratios[1] * num_samples)
        
        splits[feature] = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:]
        }
    
    return splits

def compute_scalers(
    data: dict[str, np.ndarray],
    train_indices: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    scalers = {}
    for feature_or_target, indices in train_indices.items():
        if feature_or_target not in data:
            raise ValueError(f"Feature/target '{feature_or_target}' not found in data.")
        
        if isinstance(indices, tuple):
            if np.max(indices[0]) >= data[feature_or_target].shape[0] or \
                np.max(indices[1]) >= data[feature_or_target].shape[1]:
                raise IndexError(f"Indices for feature / target '{feature_or_target}' exceed its data dimensions.")
            
            train_data = data[feature_or_target][indices[0]][ : , indices[1]]

            scalers[f"{feature_or_target}_min"] = np.min(train_data)
            scalers[f"{feature_or_target}_max"] = np.max(train_data)
            scalers[f"{feature_or_target}_mean"] = np.mean(train_data)
            scalers[f"{feature_or_target}_std"] = np.std(train_data)

        else:
            if np.max(indices) >= data[feature_or_target].shape[0]:
                raise IndexError(f"Indices for feature / target '{feature_or_target}' exceed its data dimensions.")
            
            train_data = data[feature_or_target][indices]

            scalers[f"{feature_or_target}_min"] = np.min(train_data, axis=0)
            scalers[f"{feature_or_target}_max"] = np.max(train_data, axis=0)
            scalers[f"{feature_or_target}_mean"] = np.mean(train_data, axis=0)
            scalers[f"{feature_or_target}_std"] = np.std(train_data, axis=0)

    return scalers

def generate_version_hash(raw_data_path: str | Path, problem_config: dict) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(Path(raw_data_path).name.encode())
    problem_config_yaml = yaml.safe_dump(problem_config['splitting'] | problem_config['data_labels'], sort_keys=True)
    hash_obj.update(problem_config_yaml.encode())
    return hash_obj.hexdigest()[:8]

def save_artifacts(
    output_dir: Path,
    data: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    scalers: dict[str, np.ndarray],
    shapes: dict[str, tuple[int]],
    config: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data arrays
    np.savez(output_dir / 'data.npz', **data)

    flat_splits = {}
    for feature, feature_splits in splits.items():
        for split_type, indices in feature_splits.items():
            flat_splits[f"{feature.upper()}_{split_type}"] = indices
            
    np.savez(output_dir / 'split_indices.npz', **flat_splits)
    
    np.savez(output_dir / 'scalers.npz', **scalers)
    
    metadata = {
        'dataset_version': output_dir.name.split('_')[-1],
        'creation_date': datetime.now().isoformat(),
        'raw_data_source': str(config['raw_data_path']),
        'split_ratios': config['splitting']['ratios'],
        'split_seed': config['splitting']['seed'],
        'features': config['data_labels']['features'],
        'targets': config['data_labels']['targets'],
        'input_functions': config['input_function_labels'],
        'coordinates': config['coordinate_keys'],
        'shapes': shapes
    }
    
    with (output_dir / 'metadata.yaml').open('w') as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

def update_version_registry(processed_dir: Path, config: dict) -> None:
    """Maintain a global registry of dataset versions."""
    registry_path = Path('data/versions.yaml')
    
    registry = {}
    if registry_path.exists():
        with registry_path.open() as f:
            registry = yaml.safe_load(f) or {}
    dataset_name = config['problem']
    entry = {
        'path': str(processed_dir),
        'created': datetime.now().isoformat(),
        'hash': processed_dir.name.split('_')[-1],
        'config_snapshot': {
            'splitting': config['splitting'],
            'data_labels': config['data_labels']
        }
    }
    
    registry.setdefault(dataset_name, []).append(entry)
    
    with registry_path.open(mode='w') as f:
        yaml.safe_dump(data=registry, stream=f, sort_keys=True)

