import torch

ERROR_METRICS = {
    'vector_l1' : torch.linalg.norm(ord=1),
    'vector_l2' : torch.linalg.norm(ord=2),
    'vector_linf' : torch.linalg.norm(ord=float('inf')),
    'matrix_fro' : torch.linalg.norm(ord='fro'),
    'matrix_nuc' : torch.linalg.norm(ord='nuc'),
    'matrix_l1' : torch.linalg.norm(ord=1),
    'matrix_largest_svd' : torch.linalg.norm(ord=2),
    'matrix_linf' : torch.linalg.norm(ord=float('inf')),
    'scalar' : torch.abs
}