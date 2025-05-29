import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, activation, dropout_rate=0.0 ,apply_activation=True):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.apply_activation = apply_activation 
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.linear2 = torch.nn.Linear(in_features=out_features, out_features=out_features)
        self.dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else torch.nn.Identity()
        self.shortcut = None
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out += identity
        if self.apply_activation:
            out = self.activation(out)
        return out
    
class ResNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation, dropout_rates=None):
        super(ResNet, self).__init__()
        self.activation = activation
        self.blocks = torch.nn.ModuleList()
        layers = [input_dim] + hidden_layers + [output_dim]
        n_blocks = len(layers) - 1

        if dropout_rates is None:
            dropout_rates = [0.0] * n_blocks
        
        if len(dropout_rates) != n_blocks:
            raise ValueError(f"Expected {n_blocks} dropout rates, got {len(dropout_rates)}")

        for i in range(n_blocks):
            in_features = layers[i]
            out_features = layers[i + 1]
            dropout_rate = dropout_rates[i]
            if i == n_blocks - 1:
                self.blocks.append(module=ResidualBlock(
                    in_features=in_features,
                    out_features=out_features, 
                    activation=activation,
                    dropout_rate=dropout_rate,
                    apply_activation=False)
                )
            else:
                self.blocks.append(module=ResidualBlock(
                    in_features=in_features, 
                    out_features=out_features, 
                    activation=activation, 
                    dropout_rate=dropout_rate,
                    apply_activation=True)
                )


    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out