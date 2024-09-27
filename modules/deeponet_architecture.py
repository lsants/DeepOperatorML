import torch
import torch.nn as nn

class FNNDeepOnet(nn.Module):
    def __init__(self, branch_layers=None, trunk_layers=None):
        super().__init__()
        self.branch_layers = self.mlp_subnet(branch_layers)
        self.trunk_layers = self.mlp_subnet(trunk_layers)
    
    def mlp_subnet(self, layers):
        net = nn.ModuleList()
        if layers is not None:
            for i in range(len(layers) - 1):
                in_features = layers[i]
                out_features = layers[i+1]
                net.append(nn.Linear(in_features, out_features))
                if i < len(layers) - 2:
                    net.append(nn.ReLU())
        return net

    def branch(self, X):
        b = X
        for layer in self.branch_layers:
            b = layer(b)
        return b

    def trunk(self, X):
        t = X
        for layer in self.trunk_layers:
            t = layer(t)
        return t
    
    def forward(self, X_branch, X_trunk):
        b = self.branch(X_branch)
        t = self.trunk(X_trunk)
        output = torch.mm(b,t.T)
        return output

class ISSDeepOnet(nn.Module):
    def __init__(self, branch_layers=None, trunk_layers=None):
        super().__init__()
        self.branch_real_layers = self.mlp_subnet(branch_layers)
        self.branch_imag_layers = self.mlp_subnet(branch_layers)
        self.trunk_layers = self.mlp_subnet(trunk_layers)

    def mlp_subnet(self, layers):
        net = nn.ModuleList()
        if layers is not None:
            for i in range(len(layers) - 1):
                in_features = layers[i]
                out_features = layers[i+1]
                net.append(nn.Linear(in_features, out_features))
                if i < len(layers) - 2:
                    net.append(nn.ReLU())
        return net

    def branch_real(self, X):
        b_real = X
        for layer in self.branch_real_layers:
            b_real = layer(b_real)
        return b_real
    
    def branch_imag(self, X):
        b_imag = X
        for layer in self.branch_imag_layers:
            b_imag = layer(b_imag)
        return b_imag

    def trunk(self, X):
        t = X
        for layer in self.trunk_layers:
            t = layer(t)
        return t
    
    def forward(self, X_branch, X_trunk):
        b_real = self.branch_real(X_branch)
        b_imag = self.branch_imag(X_branch)
        t = self.trunk(X_trunk)
        output_real = torch.mm(b_real,t.T)
        output_imag = torch.mm(b_imag,t.T)
        return (output_real, output_imag)