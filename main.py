import sys
import time
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
    parser.add_argument("--train-config", default="./configs/config_train.yaml", help="Path to training config file.")
    parser.add_argument("--test-config",  default="./configs/config_test.yaml", help="Path to testing config file.")
    parser.add_argument("--skip-train",   action="store_true",        help="Skip training and only test.")
    parser.add_argument("--skip-test",    action="store_true",        help="Skip testing and only train.")
    args = parser.parse_args()

    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.test_config) as f:
        test_cfg = yaml.safe_load(f)

    if not args.skip_train:
        model_info = train_model(args.train_config)
        test_cfg["DATAFILE"] = train_cfg["DATAFILE"]
        test_cfg["MODEL_FOLDER"] = train_cfg["MODEL_FOLDER"]
        test_cfg["MODELNAME"] = model_info["MODELNAME"]


    if not args.skip_test:
        if not args.skip_train:
            test_model(args.test_config, trained_model_config=model_info)
        else:
            test_model(args.test_config)

if __name__ == '__main__':
    main()