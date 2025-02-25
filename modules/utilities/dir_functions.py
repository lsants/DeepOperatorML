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