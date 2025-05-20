import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, activation, apply_activation=True):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.apply_activation = apply_activation 
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)
        self.shortcut = None
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += identity
        if self.apply_activation:
            out = self.activation(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, layers, activation):
        super(ResNet, self).__init__()
        self.activation = activation
        self.blocks = torch.nn.ModuleList()
        n_blocks = len(layers) - 1

        for i in range(n_blocks):
            in_features = layers[i]
            out_features = layers[i + 1]
            if i == n_blocks - 1:
                self.blocks.append(ResidualBlock(in_features, out_features, activation, apply_activation=False))
            else:
                self.blocks.append(ResidualBlock(in_features, out_features, activation, apply_activation=True))


    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out