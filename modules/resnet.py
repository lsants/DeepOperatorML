import torch

class ResNet(torch.nn.Module):
    def __init__(self, layers, activation):
        super(ResNet, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.activation = activation
        for i in range(len(layers) - 1):
            self.linears.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        num_linears = len(self.linears)

        out = self.linears[0](out)
        out = self.activation(out)
        residual = out

        for i in range(1, num_linears):
            out = self.linears[i](out)
            if i < num_linears - 1:
                out = self.activation(out)
            if i % 2 == 1:
                if residual.shape == out.shape:
                    out = out + residual
                residual = out
        return out