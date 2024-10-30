import torch

class MLP(torch.nn.Module):
    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(
                torch.nn.Linear(
                    layers[i],
                    layers[i + 1],
                )
            )
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        for i in range(len(self.linears) - 1):
            out = self.linears[i](out)
            out = self.activation(out)
        return self.linears[-1](out)