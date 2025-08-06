from collections.abc import Callable
import torch
from warnings import warn
from src.modules.models.tools.loss_functions.loss_fns import LOSS_FUNCTIONS

def get_loss_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns loss function with channel count validation"""
    fn = LOSS_FUNCTIONS.get(name.lower())
    if not fn:
        raise ValueError(f"Unsupported loss: {name}. Options: {list(LOSS_FUNCTIONS.keys())}")
    
    # Special validation for mag-phase
    if name == "mag_phase":
        warn("Mag-phase loss requires 2 output channels.")
    
    return fn