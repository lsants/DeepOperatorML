import torch.nn as nn

class FNNDeepOnet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList()
        L = len(layers)
        for layer in range(len(1,layers)):
            self.linears.append(nn.Linear(layer-1, layer))
            if layer != L - 1:
                self.linears.append(nn.Tanh())

    def branch(self, X, layers):
        b = X
        for layer in layers:
            b = layer(b)
        return b

    def trunk(self, X, layers):
        t = X
        for layer in layers:
            t = layer(t)
        return t