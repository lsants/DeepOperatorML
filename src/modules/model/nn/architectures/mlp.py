
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation, dropout_rates=None):
        super(MLP, self).__init__()
        layers = [input_dim] + hidden_layers + [output_dim]
        self.linears = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        n_layers = len(layers) - 1

        if dropout_rates is None:
            dropout_rates = [0.0] * n_layers

        if len(dropout_rates) != n_layers:
            raise ValueError(
                f"Expected {n_layers} dropout rates, got {len(dropout_rates)}")

        self.activation = activation

        for i in range(n_layers):
            self.linears.append(
                module=torch.nn.Linear(
                    in_features=layers[i], out_features=layers[i + 1],)
            )
            self.dropouts.append(
                torch.nn.Dropout(
                    dropout_rates[i]) if dropout_rates[i] > 0 else torch.nn.Identity()
            )

    def forward(self, inputs):
        out = inputs
        n_layers = len(self.linears)
        for i in range(n_layers - 1):
            out = self.linears[i](out)
            out = self.activation(out)
            out = self.dropouts[i](out)
        out = self.linears[n_layers - 1](out)
        return self.dropouts[n_layers - 1](out)
