from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class BiasConfig:
    num_channels: int
    precomputed_mean: torch.Tensor | None = None
    precomputed_mean_shape: torch.Size | None = None
    use_zero_bias: bool=False

    @classmethod
    def setup_for_training(cls, data_cfg: dict, pod_data: dict | None = None, use_zero_bias: bool=False):
        if pod_data is not None:
            precomputed_mean = pod_data['pod_mean']
        else:
            precomputed_mean = None
        num_channels = data_cfg["shapes"][data_cfg["targets"][0]][-1]
        return cls(
            precomputed_mean=precomputed_mean,
            num_channels=num_channels,
            use_zero_bias=use_zero_bias
        )

    @classmethod
    def setup_for_inference(cls, model_cfg_dict):
        num_channels = model_cfg_dict["output"]["num_channels"]
        precomputed_mean_shape = model_cfg_dict["bias"]["precomputed_mean_shape"]
        use_zero_bias = model_cfg_dict["bias"].get("use_zero_bias", False)
        return cls(
            precomputed_mean_shape=precomputed_mean_shape,
            num_channels=num_channels,
            use_zero_bias=use_zero_bias
        )


class BiasConfigValidator:
    @staticmethod
    def validate(config: BiasConfig):
        if config.precomputed_mean_shape is not None:
            # This means we're doing inference
            config.precomputed_mean = torch.rand(config.precomputed_mean_shape)
        else:
            return
