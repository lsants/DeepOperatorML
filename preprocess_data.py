import sys
import argparse
import logging
import yaml
import importlib.util
from pathlib import Path
from src.modules.data_processing import preprocessing_helper as helper_functions
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def preprocess_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True,
                        type=str, help="Problem name")
    args = parser.parse_args()

    try:
        base_dir = Path(__file__).parent
        script_path = base_dir / 'src' / 'problems' / \
            args.problem / 'problem_dependent_preprocessing.py'
        config_path = base_dir / 'configs' / 'problems' / \
            args.problem / 'config_preprocessing.yaml'
        if not base_dir.exists():
            raise FileNotFoundError(f"Problem directory not found: {base_dir}")
        if not script_path.exists():
            raise FileNotFoundError(
                f"Preprocessing script not found: {script_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        module_name = f"problem_dependent_preprocessing"
        spec = importlib.util.spec_from_file_location(
            name=module_name, location=script_path)
        module = importlib.util.module_from_spec(spec=spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module=module)

        with open(file=config_path) as f:
            problem_config = yaml.safe_load(stream=f)

        if hasattr(module, 'run_preprocessing'):
            processed_data = module.run_preprocessing(problem_config)
        else:
            raise AttributeError(
                f"Module '{script_path}' must implement 'run_preprocessing(problem_config)'")

        dim_info = helper_functions.validate_data_structure(
            data=processed_data, config=problem_config)

        features_sample_sizes = helper_functions.get_sample_sizes(
            data=processed_data, config=problem_config)

        dataset_sizes = helper_functions.get_data_shapes(
            data=processed_data, config=problem_config)

        feature_splits = helper_functions.split_features(
            sample_sizes=features_sample_sizes,
            split_ratios=problem_config['splitting']['ratios'],
            seed=problem_config['splitting']['seed']
        )

        train_indices = {
            feature: feature_splits[feature]['train']
            for feature in problem_config['data_labels']['features']
        }

        for target in problem_config['data_labels']['targets']:
            train_indices_target_rows = feature_splits[problem_config['data_labels']
                                                       ['features'][0]]['train']
            train_indices_target_cols = feature_splits[problem_config['data_labels']
                                                       ['features'][1]]['train']
            train_indices[target] = (
                train_indices_target_rows, train_indices_target_cols)

        scalers = helper_functions.compute_scalers(
            data=processed_data,
            train_indices=train_indices
        )

        train_processed_data = processed_data[target][train_indices_target_rows]

        pod_data = helper_functions.compute_pod(
            data=train_processed_data,
            var_share=problem_config['var_share']
        )

        version_hash = helper_functions.generate_version_hash(
            raw_data_path=Path(problem_config['raw_data_path']),
            problem_config=problem_config
        )
        output_dir = Path(f"data/processed/{args.problem}/{version_hash}")

        logger.info(
            f"Processed data shape: {processed_data[problem_config['data_labels']['targets'][0]].shape}")
        logger.info(f"Saving!",)
        helper_functions.save_artifacts(
            output_dir=output_dir,
            data=processed_data,
            splits=feature_splits,
            scalers=scalers,
            pod_data=pod_data,
            shapes=dataset_sizes,
            config=problem_config
        )

        helper_functions.update_version_registry(processed_dir=output_dir,
                                                 config=problem_config)

        logger.info(f"Successfully processed dataset version: {version_hash}")
    except KeyError:
        logging.error(f"Unknown problem: {args.problem}")
        raise


if __name__ == '__main__':
    preprocess_data()
