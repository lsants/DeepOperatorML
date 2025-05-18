from importlib import metadata
import numpy as np
import yaml
from src.modules.data_processing.deeponet_dataset import DeepONetDataset
from src.modules.data_processing.deeponet_sampler import DeepONetSampler
# from src.modules.data_processing.deeponet_transformer import DeepONetTransformer
from torch.utils.data import DataLoader

data = np.load('/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin_d5230a4f/data.npz')
metadata_path = '/Users/ls/Workspace/SSI_DeepONet/data/processed/kelvin_d5230a4f/metadata.yaml'

with open(file=metadata_path) as file:
    config = yaml.safe_load(file)

features_keys = config["FEATURES"]
targets_keys = config["TARGETS"]

xb = data[features_keys[0]]
xt = data[features_keys[1]]
g_u = data[targets_keys[0]]

print(xb.shape, xt.shape, g_u.shape)

dataset = DeepONetDataset(data=data, output_keys=targets_keys)

branch_size = len(dataset.branch_data)
trunk_size = len(dataset.trunk_data)

batch_sampler = DeepONetSampler(
    branch_size=branch_size,
    trunk_size=trunk_size,
    branch_batch_size=32,
    trunk_batch_size=64,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_sampler=batch_sampler,
    collate_fn=lambda x:x[0], #
    # num_workers=4
)

for batch in batch_sampler:
    indices = batch
    data = dataset[indices]
    print(data[features_keys[0]].shape, data[features_keys[1]].shape, data[targets_keys[0]].shape)
    break

for sample in dataloader:
    print(sample[features_keys[0]].dtype, sample[features_keys[1]].dtype, sample[targets_keys[0]].dtype)
    quit()


