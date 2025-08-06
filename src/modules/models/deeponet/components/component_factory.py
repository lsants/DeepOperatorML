import torch
import inspect
from typing import TypeVar
from src.modules.models.deeponet.components.bias import Bias
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.deeponet.components.bias.config import BiasConfig, BiasConfigValidator
from src.modules.models.deeponet.components.trunk.config import TrunkConfig, TrunkConfigValidator
from src.modules.models.deeponet.components.branch.config import BranchConfig, BranchConfigValidator
from src.modules.models.deeponet.components.trunk import OrthonormalTrunk, PODTrunk

T = TypeVar('T')


class BiasFactory:
    @classmethod
    def build(cls, config: BiasConfig) -> torch.nn.Module:
        BiasConfigValidator.validate(config)
        return Bias(
            num_channels=config.num_channels,
            precomputed_mean=config.precomputed_mean,
            use_zero_bias=config.use_zero_bias
        )


class BranchFactory:
    @classmethod
    def build(cls, config: BranchConfig) -> torch.nn.Module:
        BranchConfigValidator.validate(config)
        component_class, _ = ComponentRegistry.get(
            component_type=config.component_type,
            architecture=config.architecture
        )
        # Filter valid constructor parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {
            k: v for k, v in config.__dict__.items()
            if k in sig.parameters and k != "self"
        }

        return component_class(**valid_params)


class TrunkFactory:
    @classmethod
    def build(cls, config: TrunkConfig) -> torch.nn.Module:
        if config.component_type == "orthonormal_trunk":
            inner_trunk = cls.build(config.inner_config)  # type: ignore
            basis_tensor = torch.as_tensor(config.T_matrix)
            return OrthonormalTrunk(inner_trunk, basis_tensor)  # type: ignore

        if config.component_type == "pod_trunk":
            basis = config.pod_basis
            return PODTrunk(pod_basis=basis)  # type: ignore

        TrunkConfigValidator.validate(config)
        component_class, _ = ComponentRegistry.get(
            component_type=config.component_type,
            architecture=config.architecture
        )
        # Filter valid constructor parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {
            k: v for k, v in config.__dict__.items()
            if k in sig.parameters and k != "self"
        }

        return component_class(**valid_params)
