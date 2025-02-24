import json
import pprint
import datetime
from copy import deepcopy

def print_time(text="Time"):
    string = "{}: {}".format(text, datetime.datetime.now())

    return string

def pprint_layer_dict(layers):
    layers = pprint.pformat(layers, indent=1, compact=False, sort_dicts=False)
    layers = '\t'+'\t'.join(layers.splitlines(True))
    return layers


def pretty_print_dict(d, max_list_length=5, indent=2):
    """
    Pretty print a dictionary with truncated long lists.
    
    Args:
        d (dict): Dictionary to print
        max_list_length (int): Maximum list elements to show before truncation
        indent (int): Indentation spaces for nested levels
    
    Returns:
        str: Formatted string representation
    """
    def process_value(obj):
        if isinstance(obj, dict):
            return {k: process_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) > max_list_length:
                return obj[:max_list_length] + [f"... {len(obj)-max_list_length} more elements"]
            return [process_value(item) for item in obj]
        elif isinstance(obj, (tuple, set)):
            return process_value(list(obj))
        return obj

    # Create a truncated copy for display
    truncated = process_value(deepcopy(d))
    
    return json.dumps(
        truncated,
        indent=indent,
        ensure_ascii=False,
        default=lambda o: f"<<{type(o).__name__}>>"  # Handle non-serializable objects
    )