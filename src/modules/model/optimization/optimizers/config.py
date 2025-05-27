from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class OptimizerSpec:
    """Configuration for a single optimization phase"""
    epochs: int
    optimizer_type: str  # "adam", "sgd", etc.
    learning_rate: float
    l2_regularization: float
    lr_scheduler: Optional[dict] = None  # {"type": "step", "step_size": 1000, "gamma": 0.7}

@dataclass 
class OptimizerConfig:
    """Container for all optimization phases"""
    one_step_optimizer: List[OptimizerSpec]
    multi_step_optimizer: Dict[str, List[OptimizerSpec]]  # "trunk", "branch"