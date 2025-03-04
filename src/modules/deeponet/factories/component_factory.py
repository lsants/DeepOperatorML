from typing import Any, Dict
from ..components import (
    BaseBranch,
    BaseTrunk,
    TrainableBranch,
    TrainableTrunk,
    PretrainedTrunk,
    FixedTensorTrunk,
)

def trunk_factory(config: Dict[str, Any]) -> BaseTrunk:
    """
    Creates a trunk instance based on configuration.
    Expected keys:
      - 'type': one of ['trainable', 'pretrained', 'fixed']
      - Depending on type, the config should include 'module' or 'fixed_tensor'
    """
    trunk_type = config.get("type", "trainable")
    if trunk_type == "trainable":
        return TrainableTrunk(config["module"])
    elif trunk_type == "pretrained":
        return PretrainedTrunk(config["fixed_tensor"])
    elif trunk_type == "fixed":
        return FixedTensorTrunk(config["fixed_tensor"])
    else:
        raise ValueError(f"Unknown trunk type: {trunk_type}")

def branch_factory(config: Dict[str, Any]) -> BaseBranch:
    """
    Creates a branch instance based on configuration.
    Expected keys:
      - 'type': for now, typically 'trainable'
      - 'module': a torch.nn.Module for the branch.
    """
    branch_type = config.get("type", "trainable")
    if branch_type == "trainable":
        return TrainableBranch(config["module"])
    else:
        raise ValueError(f"Unknown branch type: {branch_type}")