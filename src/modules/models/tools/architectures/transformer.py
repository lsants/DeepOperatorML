import torch
from mlp import MLP
from typing import Callable

# Batching here means processing multiple independent sequences at the same time
# Data already enters the transformer padded into max length and with the attention mask

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding_matrix = torch.nn.Parameter(torch.zeros(vocab_size, d_model))

    def forward(self, indice_sequence: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[indice_sequence] # returns (B, L, d)
        
class PositionalEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, max_length: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        encoding_matrix = torch.zeros(self.max_length, output_dim)
        position = torch.arange(self.max_length, dtype=torch.float).unsqueeze(1)
        
        frequencies = (10000 ** (torch.arange(0, output_dim, 2).float() / output_dim))

        encoding_matrix[: , 0::2] = torch.sin( frequencies * position)
        encoding_matrix[: , 1::2] = torch.cos( frequencies * position)

        self.register_buffer("positional_encoding_matrix", encoding_matrix) # implement sin/cos
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding_matrix[positions] # (B, L, d) should actually be (B, L, d) for batch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wk = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wv = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wo = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, X: torch.Tensor):
        # input X is sized (B, L, d)
        Q = self.Wq(X).view(-1, self.num_heads, self.length, self.d_head)
        K = self.Wk(X).view(-1, self.num_heads, self.length, self.d_head)
        V = self.Wv(X).view(-1, self.num_heads, self.length, self.d_head)

        raw_attention_score = Q @ K
        attention_score = torch.nn.functional.softmax(raw_attention_score /  self.d_head**0.5)
        attention_by_head = attention_score @ V
        
        return self.Wo(attention_by_head.reshape(-1, self.length, self.d_model))

class TransformerEncoder(torch.nn.Module):
    def __init__(self, length: int, d_model: int, num_heads: int, hidden_layers: list[int], activation: Callable[[torch.Tensor], torch.Tensor], max_length: int):
        super().__init__()
        # implement padding for seq length
        max_length = 512 # placeholder. don't know how to structure this
        self.padding = torch.nn.functional.pad
        self.embedding = EmbeddingLayer(vocab_size=length, d_model=d_model)
        self.positional_encoding = PositionalEncoderLayer(input_dim=length, output_dim=d_model, max_length=max_length)
        self.multi_head_attention = MultiHeadAttention(length=max_length, d_model=d_model, num_heads=num_heads)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.FNN = MLP(
            input_dim=d_model,
            output_dim=d_model,
            hidden_layers=hidden_layers, 
            activation=activation
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        L = len(sequence)
        X = self.embedding(sequence)
        X = X + self.positional_encoding(torch.arange(0, L))
        X_attention = self.multi_head_attention(X)
        X = self.layer_norm(X_attention + X)
        X = self.layer_norm(self.FNN(X) + X)
        return X
    
import torch
from typing import Callable, Optional

# --- helper ----------------------------------------------------
def generate_subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Upper-triangular (1 on forbidden positions) causal mask.
    Returns shape (size, size) suitable for broadcasting to (B, h, L, L).
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

# --- decoder block --------------------------------------------
class TransformerDecoder(torch.nn.Module):
    """
    One-layer decoder block (stack N of these for a full decoder).
    Args
    ----
    d_model       : embedding dimension
    num_heads     : number of attention heads
    hidden_layers : list with hidden sizes for the FFN (e.g. [4*d_model])
    activation    : callable, e.g. torch.nn.GELU()
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_layers: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()

        # masked self-attention
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # encoder-decoder (cross) attention
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # feed-forward network
        self.ffn = MLP(
            input_dim=d_model,
            output_dim=d_model,
            hidden_layers=hidden_layers,
            activation=activation,
        )

        # layer norms
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

    # -----------------------------------------
    def forward(
        self,
        tgt: torch.Tensor,                 # (B, L_tgt, d_model)
        memory: torch.Tensor,              # (B, L_src, d_model) <- encoder output
        tgt_padding_mask: Optional[torch.Tensor] = None,   # (B, L_tgt)
        memory_padding_mask: Optional[torch.Tensor] = None # (B, L_src)
    ) -> torch.Tensor:
        
        B, L_tgt, _ = tgt.shape

        causal_mask = generate_subsequent_mask(L_tgt, tgt.device)  # (L_tgt, L_tgt)
        if tgt_padding_mask is not None:
            pad_mask = tgt_padding_mask[:, None, None, :]
            attn_mask = causal_mask[None, None, :, :] | pad_mask
        else:
            attn_mask = causal_mask[None, None, :, :]  # broadcast to (B, h, L, L)

        x = self.self_attn(tgt, tgt, tgt, mask=attn_mask)
        tgt = self.norm1(tgt + x)

        if memory_padding_mask is not None: # (B, 1, 1, L_src)
            mem_mask = memory_padding_mask[:, None, None, :]
        else:
            mem_mask = None

        x = self.cross_attn(tgt, memory, memory, mask=mem_mask)
        tgt = self.norm2(tgt + x)
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt
