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

    config['MODEL_NAME'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + config['MODEL_NAME']

    model_name = config["MODEL_NAME"]
    training_strategy = config["TRAINING_STRATEGY"].lower()
    output_handling = config["OUTPUT_HANDLING"].lower()
    model_name += "_" + training_strategy
    model_name += "_" + output_handling
    model_name += "_" + config['BRANCH_ARCHITECTURE']
    if training_strategy != 'pod':
        model_name += "_" + config['TRUNK_ARCHITECTURE']

    return model_name
    
def create_output_directories(config: Any):
    config['output_path'] = os.path.join(config["output_path"], config["model_name"])
    config['checkpoints_path'] = os.path.join(config["output_path"], "checkpoints")
    config['auxiliary_data_path'] = os.path.join(config["output_path"], 'aux')
    config['model_info_path'] = os.path.join(config["output_path"], 'config.yaml')
    config['dataset_indices_path'] = os.path.join(config["auxiliary_data_path"], f'split_indices.yaml')
    config['norm_params_path'] = os.path.join(config["auxiliary_data_path"], f'norm_params.yaml')
    config['metrics_path'] = os.path.join(config["output_path"], 'metrics')
    config['plots_path'] = os.path.join(config["output_path"], 'plots')

    paths = [config['output_path'],
            config['checkpoints_path'],
            config['auxiliary_data_path'],
            config['metrics_path'],
            config['plots_path']]
    
    for d in (paths):
        os.makedirs(d, exist_ok=True)