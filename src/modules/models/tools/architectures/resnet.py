from __future__ import annotations
from typing import Callable, Optional
import torch

def _validate_len(name: str, seq: list, expected: int) -> None:
    if len(seq) != expected:
        raise ValueError(f"Expected {expected} {name}, got {len(seq)}")

def _build_norm(out_features: int, use_bn: bool, use_ln: bool) -> torch.nn.Module:
    if use_bn and use_ln:
        raise ValueError("Choose either batch_norm or layer_norm, not both.")
    if use_bn:
        return torch.nn.BatchNorm1d(out_features)
    if use_ln:
        return torch.nn.LayerNorm(out_features)
    return torch.nn.Identity()

class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout_rate: float = 0.0,
        apply_activation: bool = True,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super(ResidualBlock, self).__init__()
        if batch_norm and layer_norm:
            raise ValueError("Choose either batch_norm or layer_norm, not both.")

        self.activation = activation
        self.apply_activation = apply_activation

        self.linear1 = torch.nn.Linear(
            in_features=in_features, out_features=out_features)
        self.linear2 = torch.nn.Linear(
            in_features=out_features, out_features=out_features)

        self.norm1 = _build_norm(
            out_features=out_features, use_bn=batch_norm, use_ln=layer_norm)
        self.norm2 = _build_norm(
            out_features=out_features, use_bn=batch_norm, use_ln=layer_norm)
        
        self.dropout = torch.nn.Dropout(
            p=dropout_rate) if dropout_rate > 0 else torch.nn.Identity()
        
        self.shortcut = None
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(
                in_features=in_features, out_features=out_features)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.linear2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.dropout(out) + identity
        if self.apply_activation:
            out = self.activation(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout_rates: Optional[list[float]] = None,
        batch_normalization: Optional[list[bool]] = None,
        layer_normalization: Optional[list[bool]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        self.activation = activation
        self.blocks = torch.nn.ModuleList()
        layers = [input_dim] + hidden_layers + [output_dim]
        n_blocks = len(layers) - 1

        dropout_rates = dropout_rates or [0.0] * n_blocks
        batch_normalization = batch_normalization or [False] * n_blocks
        layer_normalization = layer_normalization or [False] * n_blocks

        _validate_len("dropout rates", dropout_rates, n_blocks)
        _validate_len("batch_normalization flags", batch_normalization, n_blocks)
        _validate_len("layer_normalization flags", layer_normalization, n_blocks)

        for i in range(n_blocks):
            in_features = layers[i]
            out_features = layers[i + 1]
            dropout_rate = dropout_rates[i]
            batch_norm = batch_normalization[i]
            layer_norm = layer_normalization[i]
            self.blocks.append(
                module=ResidualBlock(
                        in_features=in_features,
                        out_features=out_features,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm,
                        layer_norm=layer_norm,
                        apply_activation=i != n_blocks - 1,
                    )
                )

    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out
