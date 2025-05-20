from dataclasses import dataclass
from .components.config import ComponentConfig

@dataclass
class ModelConfig:
    """Full model configuration."""
    branch: ComponentConfig
    trunk: ComponentConfig
    output_handling: str  # "single", "split", "shared_trunk"
    basis_functions: int
