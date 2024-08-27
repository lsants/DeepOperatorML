# ------------------- Define Neural Network architecture -------------------
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for L in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[L-1], layers[L]))
            if L != len(layers) - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        output = x
        for i in self.layers:
            output = i(output)
        return output
