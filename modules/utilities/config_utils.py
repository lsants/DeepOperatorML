import os
from datetime import datetime


def process_config(config):
    """
    Processes the configuration dictionary using the "PROBLEM" key to adjust the model name and paths.
    Raises a ValueError if any required key (MODELNAME, PROBLEM, TRAINING_STRATEGY, OUTPUT_HANDLING,
    DATAFILE, OUTPUT_LOG_FOLDER, or IMAGES_FOLDER) is missing or None.

    The function modifies:
      - MODELNAME: Appends the problem name and additional tags.
      - OUTPUT_LOG_FOLDER and IMAGES_FOLDER: Adjusts these by including the problem, training strategy,
        and output handling information.

    Args:
        config (dict): The raw configuration dictionary loaded from YAML.

    Returns:
        dict: The modified configuration dictionary.
    """

    config['MODELNAME'] = datetime.now().strftime("%Y%m%d") + "_" + "DeepONet"

    required_keys = ["MODELNAME", "PROBLEM", "TRAINING_STRATEGY", "OUTPUT_HANDLING",
                     "MODEL_FOLDER"]
    for key in required_keys:
        if key not in config or config[key] is None:
            raise ValueError(f"Missing required configuration key: {key}")

    problem = config["PROBLEM"].lower()

    model_name = config["MODELNAME"]
    if problem:
        model_name += "_" + problem
    training_strategy = config["TRAINING_STRATEGY"]
    if training_strategy:
        model_name += "_" + training_strategy.lower()
    if training_strategy == 'two_step':
        model_name += '_' + config['TRUNK_DECOMPOSITION']
    if config["LOSS_FUNCTION"] != 'mse':
        model_name += f"_{config['LOSS_FUNCTION']}"
    if config.get("INPUT_NORMALIZATION", False):
        model_name += "_in"
    if config.get("OUTPUT_NORMALIZATION", False):
        model_name += "_out"
    if config.get("INPUT_NORMALIZATION", False) or config.get("OUTPUT_NORMALIZATION", False):
        model_name += "_norm"
    if config.get("TRUNK_FEATURE_EXPANSION", False):
        model_name += "_trunkexp"
    output_handling = config["OUTPUT_HANDLING"].lower()
    if "share_trunk" in output_handling:
        model_name += "_single_basis"
    if "share_branch" in output_handling:
        model_name += "_mult_basis"
    elif "split_networks" in output_handling:
        model_name += "_split"

    config["MODELNAME"] = model_name

    model_folder = config["MODEL_FOLDER"]
    training_strategy_str = config["TRAINING_STRATEGY"]
    output_handling_str = config["OUTPUT_HANDLING"]
    config["MODEL_FOLDER"] = os.path.join(
        model_folder, problem, training_strategy_str, output_handling_str, model_name) + os.sep
    config["IMAGES_FOLDER"] = os.path.join(
        config["MODEL_FOLDER"], 'images') + os.sep

    return config
