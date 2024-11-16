import torch
import time
import numpy as np
from tqdm.auto import tqdm
from modules import dir_functions
from modules import preprocessing as ppr
from modules.deeponet import DeepONet
from modules.deeponet_two_step import DeepONetTwoStep
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
denormalize_g_u_real = ppr.Denormalize(g_u_real_min, g_u_real_max)
denormalize_g_u_imag = ppr.Denormalize(g_u_imag_min, g_u_imag_max)

# ------------------------------------ Initialize model -----------------------------
trunk_expansion_dim = p['TRUNK_EXPANSION_FEATURES_NUMBER']
u_dim = p["BRANCH_INPUT_SIZE"]
x_dim = p["TRUNK_INPUT_SIZE"]
n_outputs = p['N_OUTPUTS']
hidden_B = p['BRANCH_HIDDEN_LAYERS']
hidden_T = p['TRUNK_HIDDEN_LAYERS']
trunk_output_size = p['TRUNK_OUTPUT_SIZE']
num_basis = p["BASIS_FUNCTIONS"]

if not p['TWO_STEP_TRAINING']:
    num_basis = trunk_output_size

if p['TRUNK_FEATURE_EXPANSION']:
    x_dim += 4 * trunk_expansion_dim

layers_B = [u_dim] + hidden_B + [num_basis * n_outputs]
layers_T = [x_dim] + hidden_T + [trunk_output_size]

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

if p['TWO_STEP_TRAINING']:
    model = DeepONetTwoStep(branch_config=branch_config,
                    trunk_config=trunk_config,
                    A_dim=(n_outputs, trunk_output_size, num_basis)
                    ).to(device, precision)
else:
    model = DeepONet(branch_config=branch_config,
                    trunk_config=trunk_config,
                    ).to(device, precision)

# ---------------------------------- Initializing classes for training  -------------------
if not p['TWO_STEP_TRAINING']:
    optimizer = torch.optim.Adam(list(model.parameters()), lr=p["LEARNING_RATE"], weight_decay=p['L2_REGULARIZATION'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['SCHEDULER_STEP_SIZE'], gamma=p['SCHEDULER_GAMMA'])
    trainer = TrainModel(model, optimizer, scheduler)
    evaluator = TrainEvaluator(error_type)
else:
    trunk_optimizer = torch.optim.Adam(model.trunk_network.parameters(), lr=p["TRUNK_LEARNING_RATE"])
    trunk_scheduler = torch.optim.lr_scheduler.StepLR(trunk_optimizer, step_size=p['TRUNK_SCHEDULER_STEP_SIZE'], gamma=p['TRUNK_SCHEDULER_GAMMA'])
    trunk_trainer = TrainModel(model, trunk_optimizer, scheduler=trunk_scheduler, training_phase='trunk')
    trunk_evaluator = TrainEvaluator(error_type)
    
    branch_optimizer = torch.optim.Adam(model.branch_network.parameters(), lr=p["BRANCH_LEARNING_RATE"])
    branch_scheduler = torch.optim.lr_scheduler.StepLR(branch_optimizer, step_size=p['BRANCH_SCHEDULER_STEP_SIZE'], gamma=p['BRANCH_SCHEDULER_GAMMA'])
    branch_trainer = TrainModel(model, branch_optimizer, scheduler=branch_scheduler, training_phase='branch')
    branch_evaluator = TrainEvaluator(error_type)

saver = Saver(model_name, model_folder, data_out_folder, fig_folder)
best_model = None

epochs = p['N_EPOCHS']
niter_per_train_epoch = len(train_dataloader)
niter_per_val_epoch = len(val_dataloader)
best_avg_error_real = float('inf')

# ----------------------------------------- Train loop (2 step) ---------------------------------
start_time = time.time()

if p['TWO_STEP_TRAINING']:

    trunk_epochs = p['TRUNK_TRAIN_EPOCHS']
    branch_epochs = p['BRANCH_TRAIN_EPOCHS']

    # ------------------------------------------- Trunk training -------------------------------------
    
    model.set_training_phase('trunk')
    model.freeze_branch()
    model.unfreeze_trunk()
    full_batch_train = train_dataset[:]
    full_batch_val = val_dataset[:]

    for epoch in tqdm(range(trunk_epochs)):
        trunk_epoch_train_loss = 0
        trunk_epoch_train_error_real = 0
        trunk_epoch_train_error_imag = 0

        trunk_epoch_val_loss = 0
        trunk_epoch_val_error_real = 0
        trunk_epoch_val_error_imag = 0

        full_batch_train['xt'] = xt
        if p['INPUT_NORMALIZATION']:
            full_batch_train = {key: (normalize_trunk(value) if key == 'xt' \
                            else value)
                    for key, value in full_batch_train.items()}
        if p['OUTPUT_NORMALIZATION']:
            full_batch_train = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                            else normalize_g_u_imag(value) if key == 'g_u_imag'\
                            else value)
                    for key, value in full_batch_train.items()}
        if p['TRUNK_FEATURE_EXPANSION']:
            full_batch_train = {key: (ppr.trunk_feature_expansion(value, trunk_expansion_dim) if key == 'xt' else value)
                        for key, value in full_batch_train.items()}
        trunk_train_outputs = trunk_trainer(full_batch_train)
        trunk_epoch_train_loss += trunk_train_outputs['loss']
        if p['OUTPUT_NORMALIZATION']:
            basis_pred_real = denormalize_g_u_real(trunk_train_outputs['pred_real'])
            g_u_real = denormalize_g_u_real(full_batch_train['g_u_real'])
            basis_pred_imag = denormalize_g_u_imag(trunk_train_outputs['pred_imag'])
            g_u_imag = denormalize_g_u_imag(full_batch_train['g_u_imag'])
        trunk_epoch_train_error_real = trunk_evaluator.compute_error(g_u_real,
                                                            basis_pred_real)
        trunk_epoch_train_error_imag = trunk_evaluator.compute_error(g_u_imag,
                                                            basis_pred_imag)

        epoch_learning_rate = trunk_scheduler.get_last_lr()[-1]
        if p['TRUNK_CHANGE_OPTIMIZER']:
            if epoch == p['TRUNK_CHANGE_AT_EPOCH']:
                trunk_trainer.optimizer = torch.optim.Adam(list(model.trunk_network.parameters()) + [model.A_real, model.A_imag],
                                                            lr=trunk_scheduler.get_last_lr()[-1])
                p['TRUNK_LR_SCHEDULING'] = False
        if p['TRUNK_LR_SCHEDULING']:
            trunk_scheduler.step()

        trunk_evaluator.store_epoch_train_loss(trunk_epoch_train_loss)
        trunk_evaluator.store_epoch_train_real_error(trunk_epoch_train_error_real)
        trunk_evaluator.store_epoch_train_imag_error(trunk_epoch_train_error_imag)
        trunk_evaluator.store_epoch_learning_rate(epoch_learning_rate)

        full_batch_val['xt'] = xt
        if p['INPUT_NORMALIZATION']:
            full_batch_val = {key: (normalize_trunk(value) if key == 'xt' \
                            else value)
                    for key, value in full_batch_val.items()}
        if p['OUTPUT_NORMALIZATION']:
            full_batch_val = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                            else normalize_g_u_imag(value) if key == 'g_u_imag'\
                            else value)
                    for key, value in full_batch_val.items()}
        if p['TRUNK_FEATURE_EXPANSION']:
            full_batch_val = {key: (ppr.trunk_feature_expansion(value, trunk_expansion_dim) if key == 'xt' else value)
                        for key, value in full_batch_val.items()}
        trunk_val_outputs = trunk_trainer(full_batch_val, val=True)
        trunk_epoch_val_loss += trunk_val_outputs['loss']
        if p['OUTPUT_NORMALIZATION']:
            basis_pred_real = denormalize_g_u_real(trunk_val_outputs['pred_real'])
            g_u_real = denormalize_g_u_real(full_batch_val['g_u_real'])
            basis_pred_imag = denormalize_g_u_imag(trunk_val_outputs['pred_imag'])
            g_u_imag = denormalize_g_u_imag(full_batch_val['g_u_imag'])
        trunk_epoch_val_error_real = trunk_evaluator.compute_error(g_u_real,
                                                            basis_pred_real)
        trunk_epoch_val_error_imag = trunk_evaluator.compute_error(g_u_imag,
                                                            basis_pred_imag)

        trunk_evaluator.store_epoch_val_loss(trunk_epoch_val_loss)
        trunk_evaluator.store_epoch_val_real_error(trunk_epoch_val_error_real)
        trunk_evaluator.store_epoch_val_imag_error(trunk_epoch_val_error_imag)

    # ---------------- ---------------------------------- Trunk Decomposition ---------------------------------------------------------------------------
    
    with torch.no_grad():
        trunk_out = model.trunk_network(full_batch_val['xt'])
        if p['TRUNK_DECOMPOSITION'] == 'qr':
            _, R = torch.linalg.qr(trunk_out)
        if p['TRUNK_DECOMPOSITION'] == 'svd':
            pass
        model.trunk_basis = R
    
    # ---------------- ---------------------------------- Branch Training ---------------------------------------------------------------------------

    model.set_training_phase('branch')
    model.freeze_trunk()
    model.unfreeze_branch()

    for epoch in tqdm(range(branch_epochs)):
        branch_epoch_train_loss = 0
        branch_epoch_train_error_real = 0
        branch_epoch_train_error_imag = 0

        branch_epoch_val_loss = 0
        branch_epoch_val_error_real = 0
        branch_epoch_val_error_imag = 0

        for branch_batch in train_dataloader:
            if p['INPUT_NORMALIZATION']:
                branch_batch = {key: (normalize_branch(value) if key == 'xb' \
                            else value)
                    for key, value in branch_batch.items()}
            if p['OUTPUT_NORMALIZATION']:
                branch_batch = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                                else normalize_g_u_imag(value) if key == 'g_u_imag'\
                                else value)
                        for key, value in branch_batch.items()}
            branch_batch_train_outputs = branch_trainer(branch_batch)
            branch_epoch_train_loss += branch_batch_train_outputs['loss']
            if p['OUTPUT_NORMALIZATION']:
                branch_batch_pred_real = denormalize_g_u_real(branch_batch_train_outputs['pred_real'])
                branch_batch_g_u_real = denormalize_g_u_real(branch_batch['g_u_real'])
                branch_batch_pred_imag = denormalize_g_u_imag(branch_batch_train_outputs['pred_imag'])
                branch_batch_g_u_imag = denormalize_g_u_imag(branch_batch['g_u_imag'])
            branch_batch_train_error_real = branch_evaluator.compute_error(branch_batch_g_u_real,
                                                                   branch_batch_pred_real)
            branch_batch_train_error_imag = branch_evaluator.compute_error(branch_batch_g_u_imag,
                                                                   branch_batch_pred_imag)
            branch_epoch_train_error_real += branch_batch_train_error_real
            branch_epoch_train_error_imag += branch_batch_train_error_imag

        branch_epoch_learning_rate = branch_scheduler.get_last_lr()[-1]
        if p['BRANCH_CHANGE_OPTIMIZER']:
            if epoch == p['BRANCH_CHANGE_AT_EPOCH']:
                branch_trainer.optimizer = torch.optim.Adam(list(model.branch_network.parameters()),
                                                            lr=branch_scheduler.get_last_lr()[-1])
                p['BRANCH_LR_SCHEDULING'] = False
        if p['BRANCH_LR_SCHEDULING']:
            branch_scheduler.step()

        branch_avg_epoch_train_loss = branch_epoch_train_loss / niter_per_train_epoch
        branch_avg_epoch_train_error_real = branch_epoch_train_error_real / niter_per_train_epoch
        branch_avg_epoch_train_error_imag = branch_epoch_train_error_imag / niter_per_train_epoch

        branch_evaluator.store_epoch_train_loss(branch_avg_epoch_train_loss)
        branch_evaluator.store_epoch_train_real_error(branch_avg_epoch_train_error_real)
        branch_evaluator.store_epoch_train_imag_error(branch_avg_epoch_train_error_imag)
        
        branch_evaluator.store_epoch_learning_rate(branch_epoch_learning_rate)
        for branch_batch in val_dataloader:
            if p['INPUT_NORMALIZATION']:
                branch_batch = {key: (normalize_branch(value) if key == 'xb' \
                            else value)
                    for key, value in branch_batch.items()}
            if p['OUTPUT_NORMALIZATION']:
                branch_batch = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                                else normalize_g_u_imag(value) if key == 'g_u_imag'\
                                else value)
                        for key, value in branch_batch.items()}
            branch_batch_val_outputs = branch_trainer(branch_batch)
            branch_epoch_val_loss += branch_batch_val_outputs['loss']
            if p['OUTPUT_NORMALIZATION']:
                branch_batch_pred_real = denormalize_g_u_real(branch_batch_val_outputs['pred_real'])
                branch_batch_g_u_real = denormalize_g_u_real(branch_batch['g_u_real'])
                branch_batch_pred_imag = denormalize_g_u_imag(branch_batch_val_outputs['pred_imag'])
                branch_batch_g_u_imag = denormalize_g_u_imag(branch_batch['g_u_imag'])
            branch_batch_val_error_real = branch_evaluator.compute_error(branch_batch_g_u_real,
                                                                   branch_batch_pred_real)
            branch_batch_val_error_imag = branch_evaluator.compute_error(branch_batch_g_u_imag,
                                                                   branch_batch_pred_imag)
            branch_epoch_val_error_real += branch_batch_val_error_real
            branch_epoch_val_error_imag += branch_batch_val_error_imag

        branch_epoch_learning_rate = branch_scheduler.get_last_lr()[-1]
        branch_avg_epoch_val_loss = branch_epoch_val_loss / niter_per_val_epoch
        branch_avg_epoch_val_error_real = branch_epoch_val_error_real / niter_per_val_epoch
        branch_avg_epoch_val_error_imag = branch_epoch_val_error_imag / niter_per_val_epoch

        branch_evaluator.store_epoch_val_loss(branch_avg_epoch_val_loss)
        branch_evaluator.store_epoch_val_real_error(branch_avg_epoch_val_error_real)
        branch_evaluator.store_epoch_val_imag_error(branch_avg_epoch_val_error_imag)

    if branch_avg_epoch_val_error_real < best_avg_error_real:
        best_avg_error_real = branch_avg_epoch_val_error_real
        best_model = model.state_dict()


# ----------------------------------------- Train loop (regular) ---------------------------------
else:
    for epoch in tqdm(range(epochs), colour='GREEN'):
        epoch_train_loss = 0
        epoch_train_error_real = 0
        epoch_train_error_imag = 0

        epoch_val_loss = 0
        epoch_val_error_real = 0
        epoch_val_error_imag = 0

        for batch in train_dataloader:
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
                batch = {key: (ppr.trunk_feature_expansion(value, trunk_expansion_dim) if key == 'xt' else value)
                            for key, value in batch.items()}
            batch_train_outputs = trainer(batch)
            epoch_train_loss += batch_train_outputs['loss']
            if p['OUTPUT_NORMALIZATION']:
                batch_pred_real = denormalize_g_u_real(batch_train_outputs['pred_real'])
                batch_g_u_real = denormalize_g_u_real(batch['g_u_real'])
                batch_pred_imag = denormalize_g_u_imag(batch_train_outputs['pred_imag'])
                batch_g_u_imag = denormalize_g_u_imag(batch['g_u_imag'])
            batch_train_error_real = evaluator.compute_error(batch_g_u_real,
                                                                batch_pred_real)
            batch_train_error_imag = evaluator.compute_error(batch_g_u_imag,
                                                                batch_pred_imag)
            epoch_train_error_real += batch_train_error_real
            epoch_train_error_imag += batch_train_error_imag

        if p['CHANGE_OPTIMIZER']:
            if epoch == p['CHANGE_AT_EPOCH']:
                trainer.optimizer = torch.optim.Adam(list(model.parameters()), lr=scheduler.get_last_lr()[-1])
                p['LR_SCHEDULING'] = False
        if p['LR_SCHEDULING']:
            scheduler.step()

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
                batch = {key: (ppr.trunk_feature_expansion(value, trunk_expansion_dim) if key == 'xt' else value)
                            for key, value in batch.items()}
            batch_val_outputs = trainer(batch, val=True)
            epoch_val_loss += batch_val_outputs['loss']
            if p['OUTPUT_NORMALIZATION']:
                batch_pred_real = denormalize_g_u_real(batch_val_outputs['pred_real'])
                batch_g_u_real = denormalize_g_u_real(batch['g_u_real'])
                batch_pred_imag = denormalize_g_u_imag(batch_val_outputs['pred_imag'])
                batch_g_u_imag = denormalize_g_u_imag(batch['g_u_imag'])
            batch_val_error_real = evaluator.compute_error(batch_g_u_real,
                                                                batch_pred_real)
            batch_val_error_imag = evaluator.compute_error(batch_g_u_imag,
                                                                batch_pred_imag)
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

if p["TWO_STEP_TRAINING"]:
    trunk_loss_history = trunk_evaluator.get_loss_history()
    trunk_error_history = trunk_evaluator.get_error_history()
    trunk_lr_history = trunk_evaluator.get_lr_history()
    trunk_history = {'loss' : trunk_loss_history,
            'error' : trunk_error_history,
            'learning_rate': trunk_lr_history,
            'trunk_architecture' : (p['TRUNK_ARCHITECTURE'], layers_T)
            } 
    branch_loss_history = branch_evaluator.get_loss_history()
    branch_error_history = branch_evaluator.get_error_history()
    branch_lr_history = branch_evaluator.get_lr_history()
    branch_history = {'loss' : branch_loss_history,
            'error' : branch_error_history,
            'learning_rate': branch_lr_history,
            'branch_architecture' : (p['BRANCH_ARCHITECTURE'], layers_B)
            }

training_time = {'time': end_time - start_time}
print(f"Training concluded in: {end_time - start_time} s")

# ------------------------------------ Plot --------------------------------
if not p["TWO_STEP_TRAINING"]:
    epochs_plot = [i for i in range(epochs)]
    fig = plot_training(epochs_plot, history)
else:
    trunk_epochs_plot = [i for i in range(trunk_epochs)]
    trunk_fig = plot_training(trunk_epochs_plot, trunk_history)
    branch_epochs_plot = [i for i in range(branch_epochs)]
    branch_fig = plot_training(branch_epochs_plot, branch_history)

# --------------------------- Save output -------------------------------
if not p["TWO_STEP_TRAINING"]:
    saver(model_state_dict=best_model,
          split_indices=dataset_indices,
          norm_params=norm_params,
          history=history,
          figure=fig,
          time=training_time,
          figure_suffix="history",
          time_prefix ="training")
else:
    saver(model_state_dict=best_model,
          split_indices=dataset_indices,
          norm_params=norm_params,
          time=training_time,
          time_prefix ="training")
    
    saver(history=trunk_history,
          figure=trunk_fig,
          figure_suffix="trunk_history")
    saver(history=branch_history,
          figure=branch_fig,
          figure_suffix="branch_history")

