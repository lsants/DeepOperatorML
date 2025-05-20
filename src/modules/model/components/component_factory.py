from typing import Any, Dict
from ..optimization.loss_functions.loss_factory import LossFactory
from ..training_strategies.strategy_factory import StrategyFactory
from ..nn.activation_functions.activation_factory import ActivationFactory


def trunk_factory(config: Dict[str, Any]) -> Trunk:
    """
    Creates a trunk instance based on configuration.
    Expected keys:
      - 'type': one of ['trainable', 'pretrained', 'data']
      - Depending on type, the config should include 'module' or 'fixed_tensor'
    """
    trunk_type = config.get("type", "trainable")
    if trunk_type == "trainable":
        return TrainableTrunk(config["module"])
    elif trunk_type == "pretrained":
        return PretrainedTrunk(config["fixed_tensor"])
    elif trunk_type == "data":
        return PODTrunk(**config["data"])
    else:
        raise ValueError(f"Unknown trunk type: {trunk_type}")

def branch_factory(config: Dict[str, Any]) -> Branch:
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