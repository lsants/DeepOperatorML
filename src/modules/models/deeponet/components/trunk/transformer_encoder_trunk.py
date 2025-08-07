import torch
from typing import Optional
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.tools.architectures.transformer_encoder import LearnedReadoutTransformer


@ComponentRegistry.register(component_type="neural_trunk", architecture="transformer_encoder")
class TransformerEncoderTrunk(torch.nn.Module):
    """
    Transformer-based trunk with learned linear read-out.
    Produces a (B, output_dim) vector ready for the DeepONet inner product.
    """

    def __init__(
        self,
        input_dim: int,            # vocab / token count
        output_dim: int,           # P  (must equal branch last layer)
        d_model: int,              # transformer hidden width
        num_heads: int,
        num_layers: int,
        ff_mult: int = 4,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = LearnedReadoutTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            ff_mult=ff_mult,
            max_len=max_length,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,                   # (B, L)
        padding_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:                         # (B, output_dim)
        return self.net(tokens, padding_mask)
    
