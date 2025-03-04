import torch
from ..nn.network_architectures import NETWORK_ARCHITECTURES

class NetworkFactory:
    @staticmethod
    def create_network(config: dict) -> torch.nn.Module:
        config = config.copy()
        architecture_name = config.pop('architecture').lower()
        if architecture_name not in NETWORK_ARCHITECTURES:
            raise ValueError(f"Architecture '{architecture_name}' not implemented.")
        
        constructor = NETWORK_ARCHITECTURES[architecture_name]

        required_params = {'layers', 'activation'}
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter '{param}' for architecture '{architecture_name}'.")
        return constructor(**config)