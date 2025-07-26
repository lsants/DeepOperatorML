import torch
from mlp import MLP
from typing import Callable

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.embedding_matrix = torch.nn.Linear(in_features=length, out_features=d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix(sequence)
        
class PositionalEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, max_length: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        encoding_matrix = torch.zeros(self.max_length, output_dim)
        position = torch.arange(self.max_length, dtype=torch.float).unsqueeze(1)
        
        frequencies = 1 / (2 * torch.pi) / (10000 * torch.exp(torch.arange(0, output_dim, 2).float() / output_dim))

        encoding_matrix[: , 0::2] = torch.sin(2 * torch.pi * frequencies * position)
        encoding_matrix[: , 1::2] = torch.cos(2 * torch.pi * frequencies * position)

        self.register_buffer("positional_encoding_matrix", encoding_matrix) # implement sin/cos
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding_matrix[positions]

class MultiHeadAttention(torch.nn.Module):
    #comments: sequence length must be fixed but it is mutable, we will work with a padded length here. This means we need masking to tell the model what's padding so it'll ignore it
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.Wk = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.Wv = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.Wo = torch.nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, X: torch.Tensor):
        Q = self.Wq(X).view(-1, self.d_head, self.num_heads) # view of reshape? This is wrong and should be fixed. everything is by batch
        K = self.Wk(X).view(-1, self.d_head, self.num_heads)
        V = self.Wv(X).view(-1, self.d_head, self.num_heads)

        raw_attention_score = torch.einsum('ldh,dlh->llh', Q, K)
        attention_score = torch.nn.functional.softmax(raw_attention_score /  self.d_model**0.5)
        attention_by_head = torch.einsum('llh,ldh->ldh', attention_score, V)
        
        return attention_by_head.view(-1, self.d_model) @ self.Wo


class TransformerEncoder(torch.nn.Module):
    def __init__(self, length: int, d_model: int, num_heads: int, hidden_layers: list[int], activation: Callable[[torch.Tensor], torch.Tensor], max_length: int):
        super().__init__()
        # implement padding for seq length
        max_length = 512 # placeholder. don't know how to structure this
        self.embedding = EmbeddingLayer(length=length, d_model=d_model)
        self.positional_encoding = PositionalEncoderLayer(input_dim=length, output_dim=d_model, max_length=max_length)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.FNN = MLP(
            input_dim=d_model,
            output_dim=d_model,
            hidden_layers=hidden_layers, 
            activation=activation
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        X = self.embedding(sequence)
        X = X + self.positional_encoding(X)
        A = self.multi_head_attention(X)
        X_attention = self.layer_norm(A + X)
        X = self.FNN(X_attention) + X_attention
        X = self.layer_norm(X)
        return X
    
# implement decoder