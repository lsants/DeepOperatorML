from collections.abc import Callable, Iterable
from .loss_fns import LOSS_FUNCTIONS
from typing import Any
import torch

class LossFactory:
    @staticmethod
    def get_loss_function(name: str, model_params: dict[str, Any]) -> Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]], torch.Tensor]:
        if name not in LOSS_FUNCTIONS:
            raise ValueError(
                f"Unsupported loss function: '{name}'. Supported function are: {list(LOSS_FUNCTIONS.keys())}"
            )
        if name == "mag_phase" and len(model_params["OUTPUT_KEYS"]) != 2:
            raise ValueError(f"Invalid loss function '{name}' for non-complex targets.")
        return LOSS_FUNCTIONS[name]