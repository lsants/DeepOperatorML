from dataclasses import dataclass

@dataclass
class RescalingConfig:
    embedding_dimension: int
    exponent: float

    @classmethod
    def setup_for_training(cls, train_cfg: dict):
        embedding_dimension = train_cfg["embedding_dimension"]
        exponent = train_cfg["rescaling"]["exponent"]
        return cls(
            exponent=exponent,
            embedding_dimension=embedding_dimension
        )

    @classmethod
    def setup_for_inference(cls, model_cfg_dict):
        key = "embedding_dimension" if "embedding_dimension" in model_cfg_dict["rescaling"] else "num_basis_functions"
        embedding_dimension = model_cfg_dict["rescaling"][key]
        exponent = model_cfg_dict["rescaling"]["exponent"]
        return cls(
            exponent=exponent,
            embedding_dimension=embedding_dimension
        )
