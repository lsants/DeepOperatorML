import torch
import torch.nn as nn

class FNNDeepOnet(nn.Module):
    def __init__(self, branch_layers=None, trunk_layers=None):
        super().__init__()
        
        self.branch_layers = nn.ModuleList()
        if branch_layers is not None:
            for i in range(len(branch_layers) - 1):
                in_features = branch_layers[i]
                out_features = branch_layers[i+1]
                self.branch_layers.append(nn.Linear(in_features, out_features))
                if i < len(branch_layers) - 2:
                    self.branch_layers.append(nn.ReLU())

        self.trunk_layers = nn.ModuleList()
        if trunk_layers is not None:
            for i in range(len(trunk_layers) - 1):
                in_features = trunk_layers[i]
                out_features = trunk_layers[i+1]
                self.trunk_layers.append(nn.Linear(in_features, out_features))
                if i < len(trunk_layers) - 2:
                    self.trunk_layers.append(nn.ReLU())

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