from ..optimization.loss_fns import LOSS_FUNCTIONS

class LossFactory:
    @staticmethod
    def get_loss_function(name: str, model_params: dict) -> dict[str, callable]:
        if name not in LOSS_FUNCTIONS:
            raise ValueError(
                f"Unsupported loss function: '{name}'. Supported function are: {list(LOSS_FUNCTIONS.keys())}"
            )
        if name == "mag_phase" and len(model_params["OUTPUT_KEYS"]) != 2:
            raise ValueError(f"Invalid loss function '{name}' for non-complex targets.")
        return LOSS_FUNCTIONS[name]