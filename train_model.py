import torch
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from modules import dir_functions
from modules import preprocessing as ppr
from modules.vanilla_deeponet import VanillaDeepONet
from modules.compose_transformations import Compose
from modules.greenfunc_dataset import GreenFuncDataset
from modules.training import TrainModel
from modules.model_evaluator import Evaluator
from modules.saving import Saver
from modules.plotting import plot_training

# --------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
path_to_data = p['DATAFILE']
print(f"Training data from: {path_to_data}")

# ---------------- Defining training parameters and output paths ---------------
torch.manual_seed(p['SEED'])
precision = eval(p['PRECISION'])
device = p['DEVICE']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']

# --------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=precision, device=device)

transformations = Compose([
    to_tensor_transform
])

data = GreenFuncDataset(path_to_data, transformations)
xt = data.get_trunk()

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=p['BATCH_SIZE'], shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=p['BATCH_SIZE'], shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=p['BATCH_SIZE'], shuffle=False
)

dataset_indices = {'train': train_dataset.indices,
                   'val': val_dataset.indices,
                   'test': test_dataset.indices}

total_samples = len(data)
train_size = int(p['TRAIN_PERC'] * total_samples)
val_size = int(p['VAL_PERC'] * total_samples)
test_size = total_samples - train_size - val_size

# ----------------- Setup data normalization functions ---------------
branch_norm_params = ppr.get_branch_minmax_norm_params(train_dataloader)
trunk_norm_params = data.get_trunk_normalization_params()

xb_min, xb_max = branch_norm_params['min'], branch_norm_params['max']
xt_min, xt_max = trunk_norm_params

norm_params = {'branch': {k:v.item() for k,v in branch_norm_params.items()},
               'trunk': trunk_norm_params.tolist()}

normalize_branch, normalize_trunk = ppr.Normalize(xb_min, xb_max), ppr.Normalize(xt_min, xt_max)
denormalize_branch, denormalize_trunk = ppr.Denormalize(xb_min, xb_max), ppr.Denormalize(xt_min, xt_max)

# ----------------- Initialize model -------------------
u_dim = p["BRANCH_INPUT_SIZE"]
x_dim = p["TRUNK_INPUT_SIZE"]
n_branches = p['N_BRANCHES']
hidden_B = p['BRANCH_HIDDEN_LAYERS']
hidden_T = p['TRUNK_HIDDEN_LAYERS']
G_dim = p["BASIS_FUNCTIONS"]

layers_B = [u_dim] + hidden_B + [G_dim * n_branches]
layers_T = [x_dim] + hidden_T + [G_dim]

try:
    if p['ACTIVATION_FUNCTION'].lower() == 'relu':
        activation = torch.nn.ReLU()
    elif p['ACTIVATION_FUNCTION'].lower() == 'tanh':
        activation = torch.tanh
    else:
        raise ValueError
except ValueError:
    print('Invalid activation function.')


model = VanillaDeepONet(branch_layers=layers_B,
                        trunk_layers=layers_T,
                        activation=torch.nn.ReLU()).to(device, precision)

optimizer = torch.optim.Adam(list(model.parameters()), lr=p["LEARNING_RATE"], weight_decay=p['L2_REGULARIZATION'])
error_type = p['ERROR_NORM']

# ----------------- Initializing classes for training  ----------------
trainer = TrainModel(model, optimizer)
evaluator = Evaluator(error_type)
saver = Saver(model_name, model_folder, data_out_folder, fig_folder)

epochs = p['N_EPOCHS']
niter_per_train_epoch = len(train_dataloader)
niter_per_val_epoch = len(val_dataloader)

# ------------------- Train loop --------------------------
start_time = time.time()

for epoch in tqdm(range(epochs), colour='GREEN'):
    epoch_train_loss = 0
    epoch_train_error_real = 0
    epoch_train_error_imag = 0

    epoch_val_loss = 0
    epoch_val_error_real = 0
    epoch_val_error_imag = 0

    for batch in train_dataloader:
        model.train()
        norm_batch = {key : (normalize_branch(value) if key == 'xb' else value) \
                        for key, value in batch.items()}
        norm_batch['xt'] = normalize_trunk(xt)
        batch_train_outputs = trainer(norm_batch)
        epoch_train_loss += batch_train_outputs['loss']
        batch_train_error_real = evaluator.compute_batch_error(batch_train_outputs['pred_real'],
                                                                    norm_batch['g_u_real'])
        batch_train_error_imag = evaluator.compute_batch_error(batch_train_outputs['pred_imag'],
                                                                    norm_batch['g_u_imag'])
        epoch_train_error_real += batch_train_error_real
        epoch_train_error_imag += batch_train_error_imag

    avg_epoch_train_loss = epoch_train_loss / niter_per_train_epoch
    avg_epoch_train_error_real = epoch_train_error_real / niter_per_train_epoch
    avg_epoch_train_error_imag = epoch_train_error_imag / niter_per_train_epoch

    evaluator.store_epoch_train_loss(avg_epoch_train_loss)
    evaluator.store_epoch_train_real_error(avg_epoch_train_error_real)
    evaluator.store_epoch_train_imag_error(avg_epoch_train_error_imag)

    for batch in val_dataloader:
        norm_batch = {key : (normalize_branch(value) if key == 'xb' else value) \
                        for key, value in batch.items()}
        norm_batch['xt'] = normalize_trunk(xt)
        batch_val_outputs = trainer(norm_batch, val=True)
        epoch_val_loss += batch_val_outputs['loss']
        batch_val_error_real = evaluator.compute_batch_error(batch_val_outputs['pred_real'],
                                                                    norm_batch['g_u_real'])
        batch_val_error_imag = evaluator.compute_batch_error(batch_val_outputs['pred_imag'],
                                                                    norm_batch['g_u_imag'])
        epoch_val_error_real += batch_val_error_real
        epoch_val_error_imag += batch_val_error_imag

    avg_epoch_val_loss = epoch_val_loss / niter_per_val_epoch
    avg_epoch_val_error_real = epoch_val_error_real / niter_per_val_epoch
    avg_epoch_val_error_imag = epoch_val_error_imag / niter_per_val_epoch

    evaluator.store_epoch_val_loss(avg_epoch_val_loss)
    evaluator.store_epoch_val_real_error(avg_epoch_val_error_real)
    evaluator.store_epoch_val_imag_error(avg_epoch_val_error_imag)

end_time = time.time()

loss_history = evaluator.get_loss_history()
error_history = evaluator.get_error_history()

history = {'loss' : loss_history,
           'error' : error_history}

print(f"Training concluded in: {end_time - start_time} s")

# ----------------- Plot ---------------
epochs_plot = [i for i in range(epochs)]
fig = plot_training(epochs_plot, history)

plt.show()

# ----------- Save output ------------
saver(model_state_dict=model.state_dict(),
      split_indices=dataset_indices,
      norm_params=norm_params,
      history=history,
      figure=fig)
