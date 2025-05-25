from dataclasses import dataclass

@dataclass
class RescalingConfig:
    num_basis_functions: int
    exponent: float