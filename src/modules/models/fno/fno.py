import torch

class FourierLayer(torch.nn.Module):
    pass

# Take mesh input
class FNO(torch.nn.Module):
    def __init__(self, d: int, L: int) -> None:
        "Architecture based on  'Fourier Neural Operator for Parametric Partial Differential Equations' by Zhang et al. (2021)"
        "d: embedding size; L: number of Fourier Layers"
        super().__init__()
        self.embedding = None # Lifting z(x) to dimension dz

    

    def forward(self):
        pass