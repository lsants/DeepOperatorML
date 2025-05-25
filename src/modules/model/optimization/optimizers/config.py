from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class OptimizerPhaseConfig:
    """Configuration for a single optimization phase"""
    epochs: int
    optimizer: str  # "adam", "sgd", etc.
    learning_rate: float
    l2_regularization: float
    lr_scheduler: Optional[dict] = None  # {"type": "step", "step_size": 1000, "gamma": 0.7}

@dataclass 
class OptimizerConfig:
    """Container for all optimization phases"""
    global_schedule: List[OptimizerPhaseConfig]
    phase_specific: Dict[str, List[OptimizerPhaseConfig]]  # "trunk", "branch"