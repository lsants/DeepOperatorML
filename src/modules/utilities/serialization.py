from typing import Any
from dataclasses import is_dataclass, asdict, fields
import torch

def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to safe dictionaries"""
    if is_dataclass(obj):
        # Convert dataclass to dict and process each value
        return {
            key: dataclass_to_dict(value)
            for key, value in asdict(obj).items()
        }
    elif isinstance(obj, (list, tuple)):
        # Process each item in sequences
        return type(obj)(dataclass_to_dict(item) for item in obj)
    elif isinstance(obj, dict):
        # Process each value in dictionaries
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (torch.dtype, torch.device)):
        # Convert special types to serializable forms
        return str(obj)
    else:
        # Return primitives/tensors as-is
        return obj

# 2. Enhanced reconstruction function
def dict_to_dataclass(data: Any, target_type: type) -> Any:
    """Recursively rebuild dataclasses from safe dictionaries"""
    # Handle nested dataclasses
    if is_dataclass(target_type):
        # Get field types from target dataclass
        field_types = {f.name: f.type for f in fields(target_type)}
        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                # Recursively rebuild nested structures
                kwargs[key] = dict_to_dataclass(data=value, target_type=field_types[key])
        return target_type(**kwargs)
    
    # Handle special types
    if isinstance(data, str):
        if "torch." in data:
            if "dtype" in data:
                return getattr(torch, data.split(".")[-1])
            elif "device" in data:
                return torch.device(data.split(":")[-1])
    
    # Return other types as-is
    return data