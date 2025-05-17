import os
import sys
import argparse 
import yaml
import logging
from src import train_model
from src import test_model
    
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
    parser.add_argument("--skip-train",   action="store_true",        help="Skip training and only test.")
    parser.add_argument("--skip-test",    action="store_true",        help="Skip testing and only train.")
    args = parser.parse_args()

    problem_path = os.path.join("./configs/problems/", args.problem)
    train_config_path = args.train_config_path
    problem_config_path = os.path.join(problem_path, "config_problem.yaml")
    problem_test_config_path = os.path.join(problem_path, "config_test.yaml")

    if not args.skip_train:
        experiment_config = train_model(problem_config_path, train_config_path)

    if not args.skip_test:
        with open(problem_test_config_path) as f:
            test_cfg = yaml.safe_load(f)
        if not args.skip_train:
            test_cfg["PROCESSED_DATA_PATH"] = experiment_config["PROCESSED_DATA_PATH"]
            test_cfg["OUTPUT_PATH"] = experiment_config["OUTPUT_PATH"]
            test_model(test_cfg, experiment_config)
        else:
            trained_model_config_path = os.path.join(test_cfg["OUTPUT_PATH"], 'config.yaml') 
            with open(trained_model_config_path) as f:
                trained_model_config = yaml.safe_load(f)
            test_model(test_cfg, trained_model_config)

if __name__ == '__main__':
    main()