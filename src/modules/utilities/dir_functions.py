from __future__ import annotations
import os
import yaml
import json
from typing import Any
from datetime import datetime

def load_params(file):
    with open(file, 'r') as file:
        p = yaml.safe_load(file)
    return p

def load_data_info(file):
    with open(file, 'r') as file:
        i = json.load(file)
        return i
    
def get_model_name(config: dict[str, Any]) -> str:
     
    """
    Processes the configuration dictionary using the "PROBLEM" key to adjust the model name and paths.
    Raises a ValueError if any required key (MODEL_NAME, PROBLEM, TRAINING_STRATEGY, OUTPUT_HANDLING,
    DATA_FILE, OUTPUT_LOG_FOLDER, or IMAGES_FOLDER) is missing or None.

    The function modifies:
      - MODEL_NAME: Appends the problem name and additional tags.
      - OUTPUT_LOG_FOLDER and IMAGES_FOLDER: Adjusts these by including the problem, training strategy,
        and output handling information.

    Args:
        model_config (dict): The model configuration dictionary loaded from YAML.

    Returns:
        dict: The modified configuration dictionary.
    """

    config['MODEL_NAME'] = datetime.now().strftime("%Y-%m-%d_%H-%M")

    model_name = config["MODEL_NAME"]
    training_strategy = config["TRAINING_STRATEGY"].lower()
    output_handling = config["OUTPUT_HANDLING"].lower()
    model_name += "_" + training_strategy
    model_name += "_" + output_handling
    model_name += "_" + config['BRANCH_ARCHITECTURE']
    if training_strategy != 'pod':
        model_name += "_" + config['TRUNK_ARCHITECTURE']

    return model_name
    
def create_output_directories(config: dict[str, Any]):
    config['OUTPUT_PATH'] = os.path.join(config["OUTPUT_PATH"], config["MODEL_NAME"])
    config['CHECKPOINTS_PATH'] = os.path.join(config["OUTPUT_PATH"], "checkpoints")
    config['AUXILIARY_DATA_PATH'] = os.path.join(config["OUTPUT_PATH"], 'aux')
    config['MODEL_INFO_PATH'] = os.path.join(config["OUTPUT_PATH"], 'config.yaml')
    config['DATASET_INDICES_PATH'] = os.path.join(config["AUXILIARY_DATA_PATH"], f'split_indices.yaml')
    config['NORM_PARAMS_PATH'] = os.path.join(config["AUXILIARY_DATA_PATH"], f'norm_params.yaml')
    config['METRICS_PATH'] = os.path.join(config["OUTPUT_PATH"], 'metrics')
    config['PLOTS_PATH'] = os.path.join(config["OUTPUT_PATH"], 'plots')

    paths = [config['OUTPUT_PATH'],
            config['CHECKPOINTS_PATH'],
            config['AUXILIARY_DATA_PATH'],
            config['METRICS_PATH'],
            config['PLOTS_PATH']]
    
    for d in (paths):
            os.makedirs(d, exist_ok=True)