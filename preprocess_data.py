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
    parser.add_argument("--problem", required=True, type=str, help="Problem name")
    args = parser.parse_args()

    try:
        base_dir = Path(__file__).parent
        script_path = base_dir / 'src' / 'problems' / args.problem / 'problem_dependent_preprocessing.py'
        config_path = base_dir / 'configs' / 'problems' / args.problem / 'config_problem.yaml'
        if not base_dir.exists():
            raise FileNotFoundError(f"Problem directory not found: {base_dir}")
        if not script_path.exists():
            raise FileNotFoundError(f"Preprocessing script not found: {script_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        module_name = f"problem_dependent_preprocessing"
        spec = importlib.util.spec_from_file_location(name=module_name, location=script_path)
        module = importlib.util.module_from_spec(spec=spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module=module)

        with open(file=config_path) as f:
            problem_config = yaml.safe_load(stream=f)

        # Execute preprocessing with config
        if hasattr(module, 'run_preprocessing'):
            processed_data = module.run_preprocessing(problem_config)
        else:
            raise AttributeError(f"Module '{script_path}' must implement 'run_preprocessing(problem_config)'")
        
        dim_info = helper_functions.validate_data_structure(processed_data, problem_config)
        
        # Get sample sizes for all features
        sample_sizes = helper_functions.get_sample_sizes(processed_data, problem_config)

        feature_splits = helper_functions.split_features(
            sample_sizes=sample_sizes,
            split_ratios=problem_config['SPLITTING']['RATIOS'],
            seed=problem_config['SPLITTING']['SEED']
        )
    
        train_indices = {
            feature: feature_splits[feature]['TRAIN']
            for feature in problem_config['DATA_LABELS']['FEATURES']
        }
        
        for target in problem_config['DATA_LABELS']['TARGETS']:
            # Assuming targets use the first feature's splits (branch)
            train_indices[target] = feature_splits[problem_config['DATA_LABELS']['FEATURES'][0]]['TRAIN']
        
        scalers = helper_functions.compute_scalers(
            data=processed_data,
            train_indices=train_indices
        )
        # Generate versioned output directory
        version_hash = helper_functions.generate_version_hash(
            raw_data_path=Path(problem_config['RAW_DATA_PATH']),
            problem_config=problem_config
        )
        output_dir = Path(f"data/processed/{args.problem}_{version_hash}")
        
        helper_functions.save_artifacts(
            output_dir=output_dir,
            data=processed_data,
            splits=feature_splits,
            scalers=scalers,
            config=problem_config
)
        
        # Update global registry
        helper_functions.update_version_registry(processed_dir=output_dir, 
                                config=problem_config)
        
        logger.info(f"Successfully processed dataset version: {version_hash}")
    except KeyError:
            logging.error(f"Unknown problem: {args.problem}")
            raise

if __name__ == '__main__':
    preprocess_data()