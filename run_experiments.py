import os
import sys
import time
import subprocess
import logging
import yaml

logging.basicConfig(
    # filename=f"./data/logs/{time.strftime('%Y%m%D%H%M%S')}_main.log",
    filemode='w',
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

BASE_CONFIG = 'configs/training/config_train.yaml'

overrides_list = [
    {"PROBLEM": "kelvin", 
     "DATA_FILE" : "./data/raw/kelvin_displacements.npz",
     "INPUT_FUNCTION_KEYS" : ["F", "mu", "nu"],
     "COORDINATE_KEYS" : ["x", "y", "z"],
     "OUTPUT_KEYS" : ["g_u"],
     "TRAINING_STRATEGY": "standard",
     "OUTPUT_HANDLING": "single_output",
     "TRUNK_FEATURE_EXPANSION": 0
     },


]

def run_experiments():
    for i, overrides in enumerate(overrides_list, start=1):
        logger.info(
            f"\n======================== RUN {i} with overrides: {overrides} =======================\n")

        with open(BASE_CONFIG, 'r') as file:
            config_data = yaml.safe_load(file)

        for key, value in overrides.items():
            config_data[key] = value

        temp_config_path = f"temp_config_{i}.yaml"
        with open(temp_config_path, 'w') as file:
            yaml.safe_dump(config_data, file)

        subprocess.run(["python3", "main.py", "--train-config",
                       temp_config_path], check=True)

        os.remove(temp_config_path)

if __name__ == "__main__":
    run_experiments()
