import os
from datetime import datetime
import yaml
import json
import numpy as np

def load_params(file):
    with open(file, 'r') as file:
        p = yaml.safe_load(file)
    return p

def load_data_info(file):
    with open(file, 'r') as file:
        i = json.load(file)
        return i