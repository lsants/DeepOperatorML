from dataclasses import dataclass


@dataclass
class RescalingConfig:
    num_basis_functions: int
    exponent: float

    @classmethod
    def setup_for_training(cls, train_cfg: dict):
        num_basis_functions = train_cfg["num_basis_functions"]
        exponent = train_cfg["rescaling"]["exponent"]
        return cls(
            exponent=exponent,
            num_basis_functions=num_basis_functions
        )

    @classmethod
    def setup_for_inference(cls, model_cfg_dict):
        num_basis_functions = model_cfg_dict["rescaling"]["num_basis_functions"]
        exponent = model_cfg_dict["rescaling"]["exponent"]
        return cls(
            exponent=exponent,
            num_basis_functions=num_basis_functions
        )
