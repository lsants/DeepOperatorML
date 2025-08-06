from src.modules.models.tools.activation_functions.activation_fns import ACTIVATION_MAP
from collections.abc import Callable
import torch

class ActivationFactory:
    @staticmethod
    def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if name not in ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation function: '{name}'. Supported \
                    function are: {list(ACTIVATION_MAP.keys())}"
            )
        
        return ACTIVATION_MAP[name]
    @staticmethod
    def has_activation(activation_key: str) -> bool:
        return activation_key in ACTIVATION_MAP