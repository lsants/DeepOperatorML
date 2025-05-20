from dataclasses import dataclass
from typing import Union, Dict, Any

@dataclass 
class StrategyConfig:
    name: str
    basis_functions: int
    var_share: float = None  # POD-specific
    trunk_epochs: int = None  # TwoStep-specific

@dataclass
class PhaseConfig:
    name: str
    epochs: int
    trainable_components: list[str]

@dataclass
class PODStrategyConfig:
    var_share: float  # Variance threshold
    use_mean: bool

@dataclass
class TwoStepStrategyConfig:
    trunk_epochs: int
    branch_epochs: int
    decomposition_type: str  # 'svd', 'pca'

StrategyConfig = Union[
    PODStrategyConfig,
    TwoStepStrategyConfig,
    Dict[str, Any]  # Fallback for other strategies
]
