from __future__ import annotations
import os
import time
import torch
import numpy as np
import logging
# from ...modules.pipe.saving import Saver
# from ..model.model_factory import ModelFactory
# from ..data_processing.deeponet_dataset import DeepONetDataset
# from typing import TYPE_CHECKING, Any
# if TYPE_CHECKING:
#     from ..model.deeponet import DeepONet

logger = logging.getLogger(__name__)

# checkpoint = torch.load(f="/Volumes/tmh/outputs/kelvin/2025-05-28_23-33-00/checkpoints/experiment.pth", weights_only=True)
transforms = torch.load(f="/Users/ls/Workspace/SSI_DeepONet/transform_state.pt", weights_only=True)
print(transforms['branch_stats'])
# print(checkpoint.keys())