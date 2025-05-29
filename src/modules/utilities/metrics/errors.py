import torch
from functools import partial

ERROR_METRICS = {
    'vector_l1': partial(torch.linalg.norm, ord=1),
    'vector_l2': partial(torch.linalg.norm, ord=2),
    'vector_linf': partial(torch.linalg.norm, ord=float('inf')),
    'matrix_fro': partial(torch.linalg.norm, ord='fro'),
    'matrix_nuc': partial(torch.linalg.norm, ord='nuc'),
    'matrix_l1': partial(torch.linalg.norm, ord=1),
    'matrix_largest_svd': partial(torch.linalg.norm, ord=2),
    'matrix_linf': partial(torch.linalg.norm, ord=float('inf')),
    'scalar': torch.abs 
}