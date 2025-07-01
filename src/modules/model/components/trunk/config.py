from __future__ import annotations
import torch
import numpy
from typing import Literal, Optional, Callable
from dataclasses import dataclass
from ..registry import ComponentRegistry


@dataclass
class TrunkConfig:
    architecture: Optional[Literal["resnet", "mlp",
                                   "chebyshev_kan", "pretrained"]]
    component_type: Literal["neural_trunk",
                            "pod_trunk", "orthonormal_trunk"] = "neural_trunk"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    # Neural architecture params
    hidden_layers: Optional[list[int]] = None
    dropout_rates: Optional[list[float]] = None
    activation: Optional[Callable | str] = None
    degree: Optional[int] = None
    # POD/Decomposed params
    inner_config: Optional[TrunkConfig] = None
    T_matrix: Optional[torch.Tensor] = None
    pod_basis: Optional[torch.Tensor] = None
    pod_mean: Optional[torch.Tensor] = None


class TrunkConfigValidator:
    @staticmethod
    def validate(config: TrunkConfig):
        if config.component_type == "orthonormal_trunk":
            TrunkConfigValidator._validate_orthonormal(config)
            return
        elif config.component_type == "pod_trunk":
            TrunkConfigValidator._validate_pod(config)
            return

        try:
            component_class, required_params = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=config.architecture
            )

            required_params = [p for p in required_params if p != "self"]
            missing = [p for p in required_params if not hasattr(config, p)]
            if missing:
                raise ValueError(
                    f"Missing required parameters for {config.architecture}: {missing}"
                )

        except ValueError as e:
            raise ValueError(f"Invalid trunk configuration: {e}") from e

        # Architecture-specific validation
        if 'kan' in str(config.architecture) and config.degree < 1:  # type: ignore
            raise ValueError("KAN requires degree >= 1")

    @staticmethod
    def _validate_orthonormal(config: TrunkConfig):
        """Special validation for orthonormal trunk configuration"""
        errors = []

        if not hasattr(config, "inner_config"):
            errors.append("Missing inner_config for orthonormal trunk")
        if not hasattr(config, "T_matrix"):
            errors.append("Missing basis matrix for orthonormal trunk")

        if hasattr(config, "T_matrix"):
            T_matrix = config.T_matrix
            if not isinstance(T_matrix, (torch.Tensor)):
                errors.append("Basis matrix must be Tensor")
            elif T_matrix is None:
                errors.append("Basis matrix cannot be empty")

        if hasattr(config, "inner_config"):
            try:
                TrunkConfigValidator.validate(config.inner_config)
            except ValueError as e:
                errors.append(f"Invalid inner trunk config: {str(e)}")

        if errors:
            raise ValueError(f"Orthonormal trunk errors: {', '.join(errors)}")

    @staticmethod
    def _validate_pod(config: TrunkConfig):
        """Special validation for POD trunk configuration"""
        errors = []
        pod_basis = config.pod_basis

        if not hasattr(config, "pod_basis"):
            errors.append("Missing pod_basis attribute for PODConfig")
        if not hasattr(config, "pod_mean"):
            errors.append("Missing pod_mean attribute for PODConfig")

        if hasattr(config, "pod_basis"):
            pod_basis = config.pod_basis
            if pod_basis is None:
                errors.append("Basis tensor is missing in PODConfig")
            elif not isinstance(pod_basis, (torch.Tensor)):
                errors.append("POD modes must be Tensor")

            elif pod_basis.shape[-1] != config.output_dim:
                errors.append(
                    "POD modes' second dimension must match the trunk's output dimension")

        if hasattr(config, "pod_mean"):
            pod_mean = config.pod_mean
            if pod_mean is None:
                errors.append("POD means cannot be empty")
            elif not isinstance(pod_mean, (torch.Tensor)):
                errors.append("POD means must be Tensor")

        if errors:
            raise ValueError(f"POD trunk errors: {', '.join(errors)}")
