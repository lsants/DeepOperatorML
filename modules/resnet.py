import torch

class ResNet(torch.nn.Module):
    def __init__(self, layers, activation):
        super(ResNet, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.activation = activation

        num_layers = len(layers)
        for layer_index in range(num_layers - 1):
            self.linears.append(
                torch.nn.Linear(
                    layers[layer_index],
                    layers[layer_index + 1]
                )
            )
    
    def forward(self, inputs):
        out = inputs
        num_layers = len(self.linears)
        for layer_index in range(num_layers - 1):
            layer_input = out
            out = self.linears[layer_index](out)
            out = self.activation(out)
            if layer_index > 0 and layer_index % 2 == 0:
                out = out + layer_input
        return self.linears[-1](out)