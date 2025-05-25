import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super(MLP, self).__init__()
        hidden_layers = [input_dim] + hidden_layers + [output_dim]
        self.linears = torch.nn.ModuleList()
        n_hidden_layers = len(hidden_layers)
        for layer_index in range(n_hidden_layers - 1):
            self.linears.append(
                torch.nn.Linear(
                    hidden_layers[layer_index],
                    hidden_layers[layer_index + 1],
                )
            )
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        n_hidden_layers = len(self.linears)
        for layer_index in range(n_hidden_layers - 1):
            out = self.linears[layer_index](out)
            out = self.activation(out)
        return self.linears[-1](out)