import os
from datetime import datetime

def process_config(model_config):
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

    model_config['MODEL_NAME'] = datetime.now().strftime("%Y-%m-%d_%H-%M")

    model_name = model_config["MODEL_NAME"]
    training_strategy = model_config["TRAINING_STRATEGY"].lower()
    output_handling = model_config["OUTPUT_HANDLING"].lower()
    model_name += "_" + training_strategy
    model_name += "_" + output_handling
    model_name += "_" + model_config['BRANCH_ARCHITECTURE']
    if training_strategy != 'pod':
        model_name += "_" + model_config['TRUNK_ARCHITECTURE']

    model_config["MODEL_NAME"] = model_name

    return model_config
