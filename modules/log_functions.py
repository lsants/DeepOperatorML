import io
import os
import numpy as np
import datetime
import pprint

from high_performance_integration.modules.deeponet import VanillaDeepONet

def print_time(text="Time"):
    string = "{}: {}".format(text, datetime.datetime.now())

    return string

def pprint_layer_dict(layers):
    layers = pprint.pformat(layers, indent=1, compact=False, sort_dicts=False)
    layers = '\t'+'\t'.join(layers.splitlines(True))
    return layers

import torch
bl = [1,2,3]
model = VanillaDeepONet(bl, bl, torch.nn.ReLU())
print(pprint_layer_dict())