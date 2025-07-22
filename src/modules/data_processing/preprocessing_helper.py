from __future__ import annotations
import hashlib
import numpy as np
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_data_structure(data: dict, config: dict) -> dict:
    """Validate feature/target dimensions based on config relationships."""
    features = config['data_labels']['features']
    targets = config['data_labels']['targets']

    for name in features + targets:
        if name not in data:
            raise ValueError(f"Missing array '{name}' in processed data")

    for target in targets:
        target_shape = data[target].shape
        if target_shape[0] != data[features[0]].shape[0]:
            raise ValueError(
                f"Target '{target}' rows don't match '{features[0]}' samples")

        if target_shape[1] != data[features[1]].shape[0]:
            raise ValueError(
                f"Target '{target}' columns don't match '{features[1]}' samples")

    return {
        'features': {f: data[f].shape[0] for f in features},
        'targets': {t: data[t].shape for t in targets}
    }


def get_data_shapes(data: dict[str, Any], config: dict[str, Any]) -> dict[str, tuple[int]]:
    """Get dataset shapes."""
    features = config['data_labels']['features']
    targets = config['data_labels']['targets']
    data_shapes = {f: data[f].shape for f in features} | {
        t: data[t].shape for t in targets}
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
    target_label: str
) -> dict[str, np.ndarray]:
    scalers = {}
    for feature_or_target, indices in train_indices.items():
        if feature_or_target not in data:
            raise ValueError(
                f"Feature/target '{feature_or_target}' not found in data.")

        if feature_or_target == target_label:
            dim = (0, 1)
        else:
            dim = 0

        if isinstance(indices, tuple):
            if np.max(indices[0]) >= data[feature_or_target].shape[0] or \
                    np.max(indices[1]) >= data[feature_or_target].shape[1]:
                raise IndexError(
                    f"Indices for feature / target '{feature_or_target}' exceed its data dimensions.")

            train_data = data[feature_or_target][indices[0]][:, indices[1]]

            scalers[f"{feature_or_target}_min"] = np.min(train_data, axis=dim)
            scalers[f"{feature_or_target}_max"] = np.max(train_data, axis=dim)
            scalers[f"{feature_or_target}_mean"] = np.mean(
                train_data, axis=dim)
            scalers[f"{feature_or_target}_std"] = np.std(train_data, axis=dim)

        else:
            if np.max(indices) >= data[feature_or_target].shape[0]:
                raise IndexError(
                    f"Indices for feature / target '{feature_or_target}' exceed its data dimensions.")

            train_data = data[feature_or_target][indices]

            scalers[f"{feature_or_target}_min"] = np.min(train_data, axis=dim)
            scalers[f"{feature_or_target}_max"] = np.max(train_data, axis=dim)
            scalers[f"{feature_or_target}_mean"] = np.mean(
                train_data, axis=dim)
            scalers[f"{feature_or_target}_std"] = np.std(train_data, axis=dim)

    logger.info(f"Done.")
    return scalers


def compute_pod(
    data:  np.ndarray,
    var_share: float,
) -> dict[str, np.ndarray]:

    def pod_stacked_data(data: np.ndarray, var_share: float) -> tuple[np.ndarray, ...]:
        n_samples, n_space, n_channels = data.shape
        snapshots = data.swapaxes(1, 2).reshape(
            n_samples * n_channels, n_space)  # (n_samples * n_channels, n_space))

        mean_field = np.mean(snapshots, axis=0, keepdims=True)  # (1, n_space)
        centered = (snapshots - mean_field)
        W, S, Vt = np.linalg.svd(centered, full_matrices=False)
        spatial_vectors = Vt.T

        explained_variance_ratio = np.cumsum(
            S**2) / np.linalg.norm(S, ord=2)**2
        
        n_vectors = max(1, (explained_variance_ratio < var_share).sum().item())

        basis = spatial_vectors[:, : n_vectors]  # (n_space, n_vectors)

        # import matplotlib.pyplot as plt

        # i = 0
        # for v in basis.T:
        #     v = np.concatenate(
        #         (np.flip(v.reshape(40, 40), axis=0), v.reshape(40, 40)), axis=0)
        #     plt.contourf(np.flipud(v.T), cmap="viridis")
        #     plt.colorbar()
        #     plt.show()
        #     i += 1
        #     if i >= 10:
        #         break
        # quit()
        # for snap in snapshots:
        #     snap = np.concatenate(
        #         (np.flip(snap.reshape(40, 40), axis=0), snap.reshape(40, 40)), axis=0)
        #     plt.contourf(np.flipud(snap.T), cmap="viridis")
        #     plt.colorbar()
        #     plt.show()
        # quit()

        return basis, mean_field

    def pod_split_data(
        data: np.ndarray,              # (N_s, N_r*N_z, N_c)
        var_share: float
    ) -> tuple[np.ndarray, np.ndarray]:

        n_samp, n_space, n_chan = data.shape
        mean_field = np.empty((n_chan, n_space))
        vectors_list: list[np.ndarray] = []

        for c in range(n_chan):
            snapshots = data[:, :, c]                      # (N_samp, n_space)
            mean_c = snapshots.mean(axis=0)                # (n_space,)
            mean_field[c] = mean_c
            A = snapshots - mean_c                         # centred

            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            V = Vt.T                                       # (n_space, rank)

            cum_var = np.cumsum(S**2) / np.sum(S**2)
            n_vectors_c = np.searchsorted(cum_var, var_share) + 1

            logger.info(f"Channel {c + 1} has {n_vectors_c} vectors")

            # (n_space, n_vectors_c)
            vectors_c = V[:, : n_vectors_c]
            vectors_list.append(vectors_c)
        min_n_vectors = min(v.shape[1] for v in vectors_list)
        adjusted_vectors_list = [v[:, : min_n_vectors] for v in vectors_list]
        basis = np.concatenate(adjusted_vectors_list, axis=1)
        return basis, mean_field

    logger.info(f"Computing POD with stacked basis...")
    stacked_basis, stacked_mean = pod_stacked_data(
        data=data, var_share=var_share)
    logger.info(f"Done.")
    logger.info(f"Computing POD with split basis...")
    split_basis, split_mean = pod_split_data(data=data, var_share=var_share)
    logger.info(f"Done.")

    pod_data = {
        "stacked_basis": stacked_basis,
        "stacked_mean": stacked_mean,
        "split_basis": split_basis,
        "split_mean": split_mean
    }
    logger.info(f"Concluded proper orthogonal decomposition")
    return pod_data


def generate_version_hash(raw_data_path: str | Path, problem_config: dict) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(Path(raw_data_path).name.encode())
    problem_config_yaml = yaml.safe_dump(
        problem_config['splitting'] | problem_config['data_labels'], sort_keys=True, allow_unicode=True)
    hash_obj.update(problem_config_yaml.encode())
    return hash_obj.hexdigest()[:8]


def save_artifacts(
    output_dir: Path,
    data: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    scalers: dict[str, np.ndarray],
    pod_data: dict[str, np.ndarray],
    shapes: dict[str, tuple[int]],
    config: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data arrays
    np.savez(output_dir / 'data.npz', **data)

    flat_splits = {}
    for feature, feature_splits in splits.items():
        for split_type, indices in feature_splits.items():
            flat_splits[f"{feature}_{split_type}"] = indices

    np.savez(output_dir / 'split_indices.npz', **flat_splits)

    np.savez(output_dir / 'scalers.npz', **scalers)

    np.savez(output_dir / 'pod.npz', **pod_data)

    metadata = {
        'dataset_version': output_dir.name.split('_')[-1],
        'creation_date': datetime.now().isoformat(),
        'raw_data_source': str(config['raw_data_path']),
        'raw_metadata_source': str(config['raw_data_path'].replace('.npz', '.yaml')),
        'split_ratios': config['splitting']['ratios'],
        'split_seed': config['splitting']['seed'],
        'features': config['data_labels']['features'],
        'targets': config['data_labels']['targets'],
        'targets_keys': config['output_keys'],
        'targets_labels': config['output_labels'],
        'input_functions': config['input_function_labels'],
        'coordinates': config['coordinate_keys'],
        'shapes': shapes,
        'pod_var_share': config['var_share']
    }

    with (output_dir / 'metadata.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)


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

    with registry_path.open(mode='w', encoding='utf-8') as f:
        yaml.safe_dump(data=registry, stream=f,
                       sort_keys=True, allow_unicode=True)
