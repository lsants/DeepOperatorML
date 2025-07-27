from typing import Literal
from dataclasses import dataclass

@dataclass
class OutputConfig:
    handler_type: Literal["split_outputs", "shared_trunk", "shared_branch"]
    num_channels: int
    dims_adjust: bool = True

    @classmethod
    def setup_for_training(cls, train_cfg: dict, data_cfg: dict):
        num_channels = data_cfg["shapes"][data_cfg["targets"][0]][-1]
        handler_type = train_cfg["output_handling"]
        return cls(
            handler_type=handler_type,
            num_channels=num_channels
        )
    @classmethod
    def setup_for_inference(cls, model_cfg_dict):
        num_channels = model_cfg_dict["output"]["num_channels"]
        handler_type = model_cfg_dict["output"]["handler_type"]
        return cls(
            handler_type=handler_type,
            num_channels=num_channels
        )