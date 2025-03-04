from modules.deeponet.nn.activation_fns import ACTIVATION_MAP

class ActivationFactory:
    @staticmethod
    def get_activation(name: str) -> dict[str, callable]:
        if name not in ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation function: '{name}'. Supported \
                    function are: {list(ACTIVATION_MAP.keys())}"
            )
        
        return ACTIVATION_MAP[name]