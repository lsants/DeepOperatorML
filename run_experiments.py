import os
import subprocess
import logging
import yaml

logger = logging.getLogger(__name__)

BASE_CONFIG = 'configs/config_train.yaml'

overrides_list = [
    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "standard", 
     "OUTPUT_HANDLING": "single_trunk_split_branch"},

    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "standard", 
     "OUTPUT_HANDLING": "split_trunk_single_branch"},

    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "two_step", 
     "OUTPUT_HANDLING": "single_trunk_split_branch"},

    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "two_step", 
     "OUTPUT_HANDLING": "split_trunk_single_branch"},

    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "pod", 
     "OUTPUT_HANDLING": "single_trunk_split_branch"},

    {"PROBLEM": "dynamic", 
     "DATAFILE" : "./data/raw/displacements.npz", 
     "INPUT_FUNCTION_KEYS" : ["xb"],
     "COORDINATE_KEYS" : ["r", "z"],
     "OUTPUT_KEYS" : ["g_u_real", "g_u_imag"],
     "TRAINING_STRATEGY": "pod", 
     "OUTPUT_HANDLING": "split_trunk_single_branch"},
]

def run_experiments():
    for i, overrides in enumerate(overrides_list, start=1):
        logger.info(f"\n======================== RUN {i} with overrides: {overrides} =======================\n")

        with open(BASE_CONFIG, 'r') as file:
            config_data = yaml.safe_load(file)

        for key, value in overrides.items():
            config_data[key] = value

        temp_config_path = f"temp_config_{i}.yaml"
        with open(temp_config_path, 'w') as file:
            yaml.safe_dump(config_data, file)
        
        subprocess.run(["python3", "main.py", "--train-config", temp_config_path], check=True)

        os.remove(temp_config_path)

if __name__ == "__main__":
    run_experiments()