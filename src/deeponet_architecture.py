import torch
import torch.nn as nn

class FNNDeepOnet(nn.Module):
    def __init__(self, branch_layers=None, trunk_layers=None):
        super().__init__()
        self.branch_layers = self.mlp_subnet(branch_layers)
        self.trunk_layers = self.mlp_subnet(trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))
    
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
        output = torch.mm(b,t.T) + self.bias
        return output

class ISSDeepOnet(nn.Module):
    def __init__(self, branch_layers=None, trunk_layers=None):
        super().__init__()
        self.branch_real_layers = self.mlp_subnet(branch_layers)
        self.branch_imag_layers = self.mlp_subnet(branch_layers)
        self.trunk_layers = self.mlp_subnet(trunk_layers)
        self.bias = nn.Parameter(torch.zeros(4))

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
        pr = int(b_real.shape[-1]/2)
        pi = int(b_imag.shape[-1]/2)
        b_real_r, b_real_z = b_real[:, :pr], b_real[:, pr:]
        b_imag_r, b_imag_z = b_imag[:, :pi], b_imag[:, pi:]
        output_real_r = torch.mm(b_real_r,t.T) + self.bias[0]
        output_imag_r = torch.mm(b_imag_r,t.T) + self.bias[1]
        output_real_z = torch.mm(b_real_z,t.T) + self.bias[2]
        output_imag_z = torch.mm(b_imag_z,t.T) + self.bias[3]
        return (output_real_r, output_imag_r, output_real_z, output_imag_z)