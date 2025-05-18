from __future__ import annotations
import os
import yaml
import json

def load_params(file):
    with open(file, 'r') as file:
        p = yaml.safe_load(file)
    return p

def load_data_info(file):
    with open(file, 'r') as file:
        i = json.load(file)
        return i
    
def create_output_directories(training_params: dict[str, any]):
    training_params['OUTPUT_PATH'] = os.path.join(training_params["OUTPUT_PATH"], training_params["MODEL_NAME"])
    training_params['CHECKPOINTS_PATH'] = os.path.join(training_params["OUTPUT_PATH"], "checkpoints")
    training_params['AUXILIARY_DATA_PATH'] = os.path.join(training_params["OUTPUT_PATH"], 'aux')
    training_params['MODEL_INFO_PATH'] = os.path.join(training_params["OUTPUT_PATH"], 'config.yaml')
    training_params['DATASET_INDICES_PATH'] = os.path.join(training_params["AUXILIARY_DATA_PATH"], f'split_indices.yaml')
    training_params['NORM_PARAMS_PATH'] = os.path.join(training_params["AUXILIARY_DATA_PATH"], f'norm_params.yaml')
    training_params['METRICS_PATH'] = os.path.join(training_params["OUTPUT_PATH"], 'metrics')
    training_params['PLOTS_PATH'] = os.path.join(training_params["OUTPUT_PATH"], 'plots')

    paths = [training_params['OUTPUT_PATH'],
            training_params['CHECKPOINTS_PATH'],
            training_params['AUXILIARY_DATA_PATH'],
            training_params['METRICS_PATH'],
            training_params['PLOTS_PATH']]
    
    for d in (paths):
            os.makedirs(d, exist_ok=True)