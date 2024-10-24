import os
import numpy as np
import time
import yaml
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
from modules.preprocessing import preprocessing
from modules.plotting import plot_training

class MLP(nn.Module):
    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(
                nn.Linear(
                    layers[i],
                    layers[i+1],
                )
            )
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        for i in range(len(self.linears) - 1):
            out = self.linears[i](out)
            out = self.activation(out)
        return self.linears[-1](out)
    
test_branch_real = torch.rand(100).reshape(100,-1)
test_branch_imag = torch.rand(100).reshape(100,-1)
test_trunk = torch.rand(200).reshape(100,-1)

branch = MLP([1] + [100] * 3 + [20*2], nn.ReLU())
branch.load_state_dict(torch.load('./models/branch_20241023.pth', weights_only=True))

trunk = MLP([2] + [100] * 3 + [20], nn.ReLU())
trunk.load_state_dict(torch.load('./models/trunk_20241023.pth', weights_only=True))

out_b = branch(test_branch_real)

out_B_real = out_b[:,:20]
out_B_imag = out_b[:,20:]

out_t = trunk(test_trunk)

out_imag = torch.matmul(out_B_real, torch.transpose(out_t, 0, 1))
# out_real = torch.matmul(out_B_real, torch.transpose(out_t, 0, 1))

print(out_imag)