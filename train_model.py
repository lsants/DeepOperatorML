import torch
import time
import numpy as np
from tqdm.auto import tqdm
from modules import dir_functions
from modules import preprocessing as ppr
from modules.deeponet import DeepONet
from modules.compose_transformations import Compose
from modules.greenfunc_dataset import GreenFuncDataset
from modules.training import TrainModel
from modules.train_evaluator import TrainEvaluator
from modules.saving import Saver
from modules.plotting import plot_training

# --------------------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
path_to_data = p['DATAFILE']
print(f"Training data from: {path_to_data}")

# -------------------- Defining training parameters and output paths ---------------
torch.manual_seed(p['SEED'])
precision = eval(p['PRECISION'])
device = p['DEVICE']
error_type = p['ERROR_NORM']
model_name = p['MODELNAME']
model_folder = p['MODEL_FOLDER']
data_out_folder = p['OUTPUT_LOG_FOLDER']
fig_folder = p['IMAGES_FOLDER']

# ---------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=precision, device=device)

transformations = Compose([
    to_tensor_transform
])

data = np.load(path_to_data)
dataset = GreenFuncDataset(data, transformations)
xt = dataset.get_trunk()

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [p['TRAIN_PERC'], p['VAL_PERC'], p['TEST_PERC']])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=min(p['BATCH_SIZE'], len(train_dataset)), shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=min(p['BATCH_SIZE'], len(val_dataset)), shuffle=False
)

dataset_indices = {'train': train_dataset.indices,
                   'val': val_dataset.indices,
                   'test': test_dataset.indices}

# ------------------------------ Setup data normalization functions ------------------------
norm_params = ppr.get_minmax_norm_params(train_dataloader)
trunk_norm_params = dataset.get_trunk_normalization_params()

xb_min, xb_max = norm_params['xb']['min'], norm_params['xb']['max']
xt_min, xt_max = trunk_norm_params['min'], trunk_norm_params['max']
g_u_real_min, g_u_real_max = norm_params['g_u_real']['min'], norm_params['g_u_real']['max']
g_u_imag_min, g_u_imag_max = norm_params['g_u_imag']['min'], norm_params['g_u_imag']['max']

norm_params['xt'] = trunk_norm_params

normalize_branch = ppr.Normalize(xb_min, xb_max)
normalize_trunk = ppr.Normalize(xt_min, xt_max)
normalize_g_u_real = ppr.Normalize(g_u_real_min, g_u_real_max)
normalize_g_u_imag = ppr.Normalize(g_u_imag_min, g_u_imag_max)

# ------------------------------------ Initialize model -----------------------------
expansion_dim = p['EXPANSION_FEATURES_NUMBER']
u_dim = p["BRANCH_INPUT_SIZE"]
x_dim = p["TRUNK_INPUT_SIZE"]
n_branches = p['N_BRANCHES']
hidden_B = p['BRANCH_HIDDEN_LAYERS']
hidden_T = p['TRUNK_HIDDEN_LAYERS']
G_dim = p["BASIS_FUNCTIONS"]
if p['TRUNK_FEATURE_EXPANSION']:
    x_dim += 4 * expansion_dim

layers_B = [u_dim] + hidden_B + [G_dim * n_branches]
layers_T = [x_dim] + hidden_T + [G_dim]

branch_config = {
    'architecture': p['BRANCH_ARCHITECTURE'],
    'layers': layers_B,
}

trunk_config = {
    'architecture': p['TRUNK_ARCHITECTURE'],
    'layers': layers_T,
}

if p['BRANCH_ARCHITECTURE'].lower() == 'mlp' or p['BRANCH_ARCHITECTURE'].lower() == 'resnet':
    try:
        if p['BRANCH_MLP_ACTIVATION'].lower() == 'relu':
            branch_activation = torch.nn.ReLU()
        elif p['BRANCH_MLP_ACTIVATION'].lower() == 'tanh':
            branch_activation = torch.tanh
        else:
            raise ValueError
    except ValueError:
        print('Invalid activation function for branch net.')
    branch_config['activation'] = branch_activation
else:
    branch_config['degree'] = p['BRANCH_KAN_DEGREE']

if p['TRUNK_ARCHITECTURE'].lower() == 'kan':
    trunk_config['degree'] = p['TRUNK_KAN_DEGREE']
else:
    try:
        if p['TRUNK_MLP_ACTIVATION'].lower() == 'relu':
            trunk_activation = torch.nn.ReLU()
        elif p['TRUNK_MLP_ACTIVATION'].lower() == 'tanh':
            trunk_activation = torch.tanh
        else:
            raise ValueError
    except ValueError:
        print('Invalid activation function for trunk net.')
    trunk_config['activation'] = trunk_activation

model = DeepONet(branch_config=branch_config,
                        trunk_config=trunk_config).to(device, precision)

optimizer = torch.optim.Adam(list(model.parameters()), lr=p["LEARNING_RATE"], weight_decay=p['L2_REGULARIZATION'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['SCHEDULER_STEP_SIZE'], gamma=p['SCHEDULER_GAMMA'])

# ---------------------------------- Initializing classes for training  -------------------
trainer = TrainModel(model, optimizer, scheduler)
evaluator = TrainEvaluator(error_type)
saver = Saver(model_name, model_folder, data_out_folder, fig_folder)
best_model = None

epochs = p['N_EPOCHS']
niter_per_train_epoch = len(train_dataloader)
niter_per_val_epoch = len(val_dataloader)

# ----------------------------------------- Train loop ---------------------------------
start_time = time.time()

for epoch in tqdm(range(epochs), colour='GREEN'):
    epoch_train_loss = 0
    epoch_train_error_real = 0
    epoch_train_error_imag = 0

    epoch_val_loss = 0
    epoch_val_error_real = 0
    epoch_val_error_imag = 0
    best_avg_error_real = float('inf')

    for batch in train_dataloader:
        model.train()
        batch['xt'] = xt
        if p['INPUT_NORMALIZATION']:
            batch = {key: (normalize_branch(value) if key == 'xb' \
                            else normalize_trunk(value) if key == 'xt'\
                            else value)
                    for key, value in batch.items()}
        if p['OUTPUT_NORMALIZATION']:
            batch = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                            else normalize_g_u_imag(value) if key == 'g_u_imag'\
                            else value)
                    for key, value in batch.items()}
        if p['TRUNK_FEATURE_EXPANSION']:
            batch = {key: (ppr.trunk_feature_expansion(value, expansion_dim) if key == 'xt' else value)
                        for key, value in batch.items()}
        batch_train_outputs = trainer(batch)
        epoch_train_loss += batch_train_outputs['loss']
        batch_train_error_real = evaluator.compute_batch_error(batch_train_outputs['pred_real'],
                                                                    batch['g_u_real'])
        batch_train_error_imag = evaluator.compute_batch_error(batch_train_outputs['pred_imag'],
                                                                    batch['g_u_imag'])
        epoch_train_error_real += batch_train_error_real
        epoch_train_error_imag += batch_train_error_imag

    if p['LR_SCHEDULING']:
        scheduler.step()
    if p['CHANGE_OPTIMIZER']:
        if epoch == p['CHANGE_AT_EPOCH']:
            trainer.change_optimizer(torch.optim.LBFGS(list(model.parameters()), lr=scheduler.get_last_lr()[-1]))
            p['LR_SCHEDULING'] = False

    epoch_learning_rate = scheduler.get_last_lr()[-1]
    avg_epoch_train_loss = epoch_train_loss / niter_per_train_epoch
    avg_epoch_train_error_real = epoch_train_error_real / niter_per_train_epoch
    avg_epoch_train_error_imag = epoch_train_error_imag / niter_per_train_epoch

    evaluator.store_epoch_train_loss(avg_epoch_train_loss)
    evaluator.store_epoch_train_real_error(avg_epoch_train_error_real)
    evaluator.store_epoch_train_imag_error(avg_epoch_train_error_imag)
    evaluator.store_epoch_learning_rate(epoch_learning_rate)

    for batch in val_dataloader:
        batch['xt'] = xt
        if p['INPUT_NORMALIZATION']:
            batch = {key: (normalize_branch(value) if key == 'xb' \
                            else normalize_trunk(value) if key == 'xt'\
                            else value)
                    for key, value in batch.items()}
        if p['OUTPUT_NORMALIZATION']:
            batch = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                            else normalize_g_u_imag(value) if key == 'g_u_imag'\
                            else value)
                    for key, value in batch.items()}
        if p['TRUNK_FEATURE_EXPANSION']:
            batch = {key: (ppr.trunk_feature_expansion(value, expansion_dim) if key == 'xt' else value)
                        for key, value in batch.items()}
        batch_val_outputs = trainer(batch, val=True)
        epoch_val_loss += batch_val_outputs['loss']
        batch_val_error_real = evaluator.compute_batch_error(batch_val_outputs['pred_real'],
                                                                    batch['g_u_real'])
        batch_val_error_imag = evaluator.compute_batch_error(batch_val_outputs['pred_imag'],
                                                                    batch['g_u_imag'])
        epoch_val_error_real += batch_val_error_real
        epoch_val_error_imag += batch_val_error_imag

        if epoch % 1000 == 0:
            print(f"Loss for epoch {epoch}: {epoch_val_loss:.3E}")

    avg_epoch_val_loss = epoch_val_loss / niter_per_val_epoch
    avg_epoch_val_error_real = epoch_val_error_real / niter_per_val_epoch
    avg_epoch_val_error_imag = epoch_val_error_imag / niter_per_val_epoch

    evaluator.store_epoch_val_loss(avg_epoch_val_loss)
    evaluator.store_epoch_val_real_error(avg_epoch_val_error_real)
    evaluator.store_epoch_val_imag_error(avg_epoch_val_error_imag)

    if avg_epoch_val_error_real < best_avg_error_real:
        best_avg_error_real = avg_epoch_val_error_real
        best_model = model.state_dict()

end_time = time.time()

loss_history = evaluator.get_loss_history()
error_history = evaluator.get_error_history()
lr_history = evaluator.get_lr_history()

history = {'loss' : loss_history,
           'error' : error_history,
           'learning_rate': lr_history,
           'branch_architecture' : (p['BRANCH_ARCHITECTURE'], layers_B),
           'trunk_architecture' : (p['TRUNK_ARCHITECTURE'], layers_T)
           }

training_time = {'time': end_time - start_time}
print(f"Training concluded in: {end_time - start_time} s")

# ------------------------------------ Plot --------------------------------
epochs_plot = [i for i in range(epochs)]
fig = plot_training(epochs_plot, history)

# --------------------------- Save output -------------------------------
saver(model_state_dict=best_model,
      split_indices=dataset_indices,
      norm_params=norm_params,
      history=history,
      figure=fig,
      time=training_time,
      figure_suffix="history",
      time_prefix ="training")
