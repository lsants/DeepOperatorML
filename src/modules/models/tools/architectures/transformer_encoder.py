import math
import torch
import torch.nn as nn
from typing import Optional

class LearnedReadoutTransformer(nn.Module):
    """
    Encoder-only transformer  ➜  learned, weighted pooling  ➜  Linear(P).
    Output shape = (B, P).
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_dim: int,
        token_feature_dim: int = 1,
        ff_mult: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(token_feature_dim, d_model, bias=False)
        self.register_buffer("pos_emb", self._build_pe(max_len, d_model)[None, ...], persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        
        self.score = nn.Linear(d_model, 1, bias=False) # (produces α_i ∈ ℝ)
        self.out_proj = nn.Linear(d_model, output_dim, bias=False) # linear projection from context vec to size P

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,                     # (B, L)
        padding_mask: Optional[torch.Tensor] = None 
    ) -> torch.Tensor:                           # → (B, P)
        B, L = tokens.shape
        if tokens.dim() == 2:                 
            tokens = tokens.unsqueeze(-1)

        x = self.input_proj(tokens)
        x = x + self.pos_emb[:, :L]                    # (B, L, D)

        h = self.encoder(x, src_key_padding_mask=padding_mask)

        raw_scores = self.score(h).squeeze(-1)                  # (B, L)
        if padding_mask is not None:
            raw_scores = raw_scores.masked_fill(padding_mask, -1e30)
        weights = torch.softmax(raw_scores, dim=1)

        context = torch.einsum("bl, bld -> bd", weights, h)      # (B, D)
        return self.out_proj(context)                            # (B, P)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (max_len, d_model)
