import torch
from .optimizers import OPTIMIZER_MAP

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name: str, parameters, config: dict) -> torch.optim.Optimizer:
        optimizer_name_lower = optimizer_name.lower()
        if optimizer_name_lower not in OPTIMIZER_MAP:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_name}'. Supported optimizers are: {list(OptimizerFactory.OPTIMIZER_MAP.keys())}"
            )
        optimizer_class = OPTIMIZER_MAP[optimizer_name_lower]

        lr = config["LEARNING_RATE"]
        optimizer_params = {'lr': lr}

        if optimizer_name_lower == 'sgd' or optimizer_name_lower == 'rmsprop':
            momentum = config.get("MOMENTUM", 0.0)
            optimizer_params["momentum"] = momentum

        return optimizer_class(parameters, **optimizer_params)