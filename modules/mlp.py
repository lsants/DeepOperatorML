import torch

class MLP(torch.nn.Module):
    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        num_layers = len(layers)
        for layer_index in range(num_layers - 1):
            self.linears.append(
                torch.nn.Linear(
                    layers[layer_index],
                    layers[layer_index + 1],
                )
            )
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        num_layers = len(self.linears)
        for layer_index in range(num_layers - 1):
            out = self.linears[layer_index](out)
            out = self.activation(out)
        return self.linears[-1](out)