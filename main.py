import os
import sys
import argparse 
import yaml
import logging
from src import train_model
from src import test_model
from src.modules.pipe.pipeline_config import DataConfig, TrainConfig, TestConfig
from src.modules.utilities import config_validation as validation

logger = logging.getLogger(__name__)

logging.basicConfig(
    filemode='w',
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="Type problem to be solved.")
    parser.add_argument("--train-config-path", default="./configs/training/config_train.yaml", help="Path to training config file.")
    parser.add_argument("--test",   action="store_true",        help="Skip training and only test.")
    args = parser.parse_args()

    problem_path = os.path.join("./configs/problems/", args.problem)
    train_config_path = args.train_config_path
    experiment_config_path = os.path.join(problem_path, "config_experiment.yaml")
    problem_test_config_path = os.path.join(problem_path, "config_test.yaml")

    data_cfg = DataConfig.from_experiment_config(
            problem=args.problem,
            exp_cfg=yaml.safe_load(open(experiment_config_path))
        )
    train_cfg = TrainConfig.from_config_files(
            exp_cfg_path=experiment_config_path, 
            train_cfg_path=train_config_path,
            data_cfg=data_cfg
        )

    # Validate individual configs
    validation.validate_train_config(train_cfg)
    # Cross-config validation
    validation.validate_config_compatibility(data_cfg, train_cfg)

    if args.test:
        test_cfg = TestConfig.from_config_files(
            test_cfg_path=problem_test_config_path,)
        test_model(test_config=test_cfg)
    else:
        train_model(data_cfg=data_cfg, train_cfg=train_cfg)


if __name__ == '__main__':
    main()