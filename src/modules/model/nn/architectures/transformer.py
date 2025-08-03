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

class TransformerEncoder(torch.nn.Module): # TODO: Adjust for padding.
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
    
# TODO: Implement decoder
class TransformerDecoder(torch.nn.Module):
    pass