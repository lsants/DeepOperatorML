import time
import copy
import torch
import numpy as np
from tqdm.auto import tqdm
from modules import dir_functions
from modules import preprocessing as ppr
from modules.saving import Saver
from modules.plotting import plot_training
from modules.model_factory import create_model
from modules.train_evaluator import TrainEvaluator
from modules.compose_transformations import Compose
from modules.greenfunc_dataset import GreenFuncDataset
from modules.training import ModelTrainer, TwoStepTrainer

# --------------------------- Load params file ------------------------
p = dir_functions.load_params('params_model.yaml')
print(f"Training data from: {p['DATAFILE']}")

torch.manual_seed(p['SEED'])

# ---------------------------- Load dataset ----------------------
to_tensor_transform = ppr.ToTensor(dtype=eval(p['PRECISION']), device=p['DEVICE'])

transformations = Compose([
    to_tensor_transform
])

data = np.load(p['DATAFILE'])
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

p['TRAIN_INDICES'] = train_dataset.indices
p['VAL_INDICES'] = val_dataset.indices
p['TEST_INDICES'] = test_dataset.indices

# ------------------------------ Setup data normalization functions ------------------------

norm_params = ppr.get_minmax_norm_params(train_dataloader)
trunk_norm_params = ppr.get_trunk_normalization_params(xt)
norm_params['xt'] = trunk_norm_params

xb_min, xb_max = norm_params['xb']['min'], norm_params['xb']['max']
xt_min, xt_max = trunk_norm_params['min'], trunk_norm_params['max']
g_u_real_min, g_u_real_max = norm_params['g_u_real']['min'], norm_params['g_u_real']['max']
g_u_imag_min, g_u_imag_max = norm_params['g_u_imag']['min'], norm_params['g_u_imag']['max']

normalize_branch = ppr.Normalize(xb_min, xb_max)
normalize_trunk = ppr.Normalize(xt_min, xt_max)
normalize_g_u_real = ppr.Normalize(g_u_real_min, g_u_real_max)
normalize_g_u_imag = ppr.Normalize(g_u_imag_min, g_u_imag_max)
denormalize_g_u_real = ppr.Denormalize(g_u_real_min, g_u_real_max)
denormalize_g_u_imag = ppr.Denormalize(g_u_imag_min, g_u_imag_max)

p["NORMALIZATION_PARAMETERS"] = norm_params

# ------------------------------------ Initialize model -----------------------------

model, model_name = create_model(p)

p['MODELNAME'] = model_name

# ---------------------------------- Initializing classes for training  -------------------

if p['TWO_STEP_TRAINING']:
    trunk_optimizer = torch.optim.Adam(list(model.trunk_network.parameters()) + list(model.A_list),
                                        lr=p["TRUNK_LEARNING_RATE"])
    trunk_scheduler = torch.optim.lr_scheduler.StepLR(trunk_optimizer, 
                                                      step_size=p['TRUNK_SCHEDULER_STEP_SIZE'], 
                                                      gamma=p['TRUNK_SCHEDULER_GAMMA'])
    trunk_trainer = TwoStepTrainer(model, 
                                   trunk_optimizer, 
                                   scheduler=trunk_scheduler, 
                                   training_phase='trunk')
    trunk_evaluator = TrainEvaluator(p['ERROR_NORM'])
    
    branch_optimizer = torch.optim.Adam(model.branch_network.parameters(), 
                                        lr=p["BRANCH_LEARNING_RATE"])
    branch_scheduler = torch.optim.lr_scheduler.StepLR(branch_optimizer, 
                                                       step_size=p['BRANCH_SCHEDULER_STEP_SIZE'], 
                                                       gamma=p['BRANCH_SCHEDULER_GAMMA'])
    branch_trainer = TwoStepTrainer(model, 
                                    branch_optimizer, 
                                    scheduler=branch_scheduler, 
                                    training_phase='branch')
    branch_evaluator = TrainEvaluator(p['ERROR_NORM'])
else:
    optimizer = torch.optim.Adam(list(model.parameters()), lr=p["LEARNING_RATE"], weight_decay=p['L2_REGULARIZATION'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['SCHEDULER_STEP_SIZE'], gamma=p['SCHEDULER_GAMMA'])
    trainer = ModelTrainer(model, optimizer, scheduler)
    evaluator = TrainEvaluator(p['ERROR_NORM'])

saver = Saver(p['MODELNAME'], p['MODEL_FOLDER'], p['OUTPUT_LOG_FOLDER'], p['IMAGES_FOLDER'])
best_model = None

epochs = p['N_EPOCHS']
niter_per_train_epoch = len(train_dataloader)
niter_per_val_epoch = len(val_dataloader)
best_avg_error_real = float('inf')

# ----------------------------------------- POD Train loop ---------------------------------

model.freeze_trunk()

start_time = time.time()

if p['PROPER_ORTHOGONAL_DECOMPOSITION']:
    full_train_dataset_real = dataset[dataset_indices['train']]['g_u_real'].T
    full_train_dataset_imag = dataset[dataset_indices['train']]['g_u_imag'].T

    mean_function_real = full_train_dataset_real.mean(axis=0)
    mean_function_imag = full_train_dataset_imag.mean(axis=0)

    full_train_dataset_real -= mean_function_real
    full_train_dataset_imag -= mean_function_imag

    U_r, S_r , V_r = torch.linalg.svd(full_train_dataset_real)
    U_i, S_i , V_i = torch.linalg.svd(full_train_dataset_imag)
    
    mean_functions = torch.stack((mean_function_real, mean_function_imag))
    pod_basis = torch.stack((U_r, U_i))

    model.get_mean_functions(mean_functions)
    model.get_basis(pod_basis)

    full_batch_train = dataset[dataset_indices['train']]

    for epoch in tqdm(range(p['N_EPOCHS']), colour='GREEN'):
        branch_epoch_train_loss = 0
        branch_epoch_train_error_real = 0
        branch_epoch_train_error_imag = 0
        branch_epoch_learning_rate = scheduler.get_last_lr()[-1]

        branch_train_data = copy.deepcopy(full_batch_train)

        if p['INPUT_NORMALIZATION']:
            branch_train_data = {key: (normalize_branch(value) if key == 'xb' \
                        else value)
                for key, value in branch_train_data.items()}

        branch_train_outputs = trainer(branch_train_data)
        branch_epoch_train_loss = branch_train_outputs['loss']

        if p['OUTPUT_NORMALIZATION']:
            branch_pred_real = denormalize_g_u_real(branch_train_outputs['pred_real'])
            branch_g_u_real = denormalize_g_u_real(branch_train_data['g_u_real'])
            branch_pred_imag = denormalize_g_u_imag(branch_train_outputs['pred_imag'])
            branch_g_u_imag = denormalize_g_u_imag(branch_train_data['g_u_imag'])

        else:
            branch_pred_real = branch_train_outputs['pred_real']
            branch_g_u_real = branch_train_data['g_u_real']
            branch_pred_imag = branch_train_outputs['pred_imag']
            branch_g_u_imag = branch_train_data['g_u_imag']
        
        branch_epoch_train_error_real = evaluator.compute_error(branch_g_u_real,
                                                                        branch_pred_real)
        branch_epoch_train_error_imag = evaluator.compute_error(branch_g_u_imag,
                                                                        branch_pred_imag)

        if p['CHANGE_OPTIMIZER']:
            if epoch == p['CHANGE_AT_EPOCH']:
                trainer.optimizer = torch.optim.Adam(list(model.branch_network.parameters()),
                                                            lr=scheduler.get_last_lr()[-1])
                p['LR_SCHEDULING'] = False

        if p['LR_SCHEDULING']:
            scheduler.step()

        if epoch % 500 == 0:
            print(f"POD loss for epoch {epoch}: {branch_epoch_train_loss:.3E}")

        evaluator.store_epoch_train_loss(branch_epoch_train_loss)
        evaluator.store_epoch_train_real_error(branch_epoch_train_error_real.item())
        evaluator.store_epoch_train_imag_error(branch_epoch_train_error_imag.item())
        
        evaluator.store_epoch_learning_rate(branch_epoch_learning_rate)
        
    if branch_epoch_train_error_real < best_avg_error_real:
        best_avg_error_real = branch_epoch_train_error_real
        best_model_checkpoint = {
            'model_state_dict': model.state_dict(),
                'POD_basis': model.basis,
                'mean_functions': model.mean_functions
            }
        
    last_model = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }

elif p['TWO_STEP_TRAINING']:

    # ------------------------------------------- Trunk training -------------------------------------
    
    model.set_training_phase('trunk')
    model.freeze_branch()
    model.unfreeze_trunk()
    model.unfreeze_A()

    full_batch_train = dataset[dataset_indices['train']]

    for epoch in tqdm(range(p['TRUNK_TRAIN_EPOCHS']), colour='YELLOW'):
        trunk_epoch_train_loss = 0
        trunk_epoch_train_error_real = 0
        trunk_epoch_train_error_imag = 0
        epoch_learning_rate = trunk_scheduler.get_last_lr()[-1]

        full_batch_train['xt'] = xt

        trunk_training_data = copy.deepcopy(full_batch_train)

        if p['INPUT_NORMALIZATION']:
            trunk_training_data = {key: (normalize_trunk(value) if key == 'xt' \
                            else value)
                    for key, value in trunk_training_data.items()}
        if p['OUTPUT_NORMALIZATION']:
            trunk_training_data = {key: (normalize_g_u_real(value) if key == 'g_u_real' \
                            else normalize_g_u_imag(value) if key == 'g_u_imag'\
                            else value)
                    for key, value in trunk_training_data.items()}
        if p['TRUNK_FEATURE_EXPANSION']:
            trunk_training_data = {key: (ppr.trunk_feature_expansion(value, p['TRUNK_EXPANSION_FEATURES_NUMBER']) if key == 'xt' else value)
                        for key, value in trunk_training_data.items()}
            
        trunk_train_outputs = trunk_trainer(trunk_training_data)

        trunk_epoch_train_loss += trunk_train_outputs['loss']

        basis_pred_real = trunk_train_outputs['pred_real']
        g_u_real = trunk_training_data['g_u_real']
        basis_pred_imag = trunk_train_outputs['pred_imag']
        g_u_imag = trunk_training_data['g_u_imag']

        if p['OUTPUT_NORMALIZATION']:
            basis_pred_real = denormalize_g_u_real(basis_pred_real)
            g_u_real = denormalize_g_u_real(g_u_real)
            basis_pred_imag = denormalize_g_u_imag(basis_pred_imag)
            g_u_imag = denormalize_g_u_imag(g_u_imag)

        trunk_epoch_train_error_real = trunk_evaluator.compute_error(g_u_real,
                                                            basis_pred_real)
        trunk_epoch_train_error_imag = trunk_evaluator.compute_error(g_u_imag,
                                                            basis_pred_imag)
        if p['TRUNK_CHANGE_OPTIMIZER']:
            if epoch == p['TRUNK_CHANGE_AT_EPOCH']:
                trunk_trainer.optimizer = torch.optim.Adam(list(model.trunk_network.parameters()) + list(model.A_list),
                                                            lr=trunk_scheduler.get_last_lr()[-1])
                p['TRUNK_LR_SCHEDULING'] = False
        if p['TRUNK_LR_SCHEDULING']:
            trunk_scheduler.step()

        if epoch % 500 == 0:
            print(f"Trunk loss for epoch {epoch}: {trunk_epoch_train_loss:.3E}")

        trunk_evaluator.store_epoch_train_loss(trunk_epoch_train_loss)
        trunk_evaluator.store_epoch_train_real_error(trunk_epoch_train_error_real.item())
        trunk_evaluator.store_epoch_train_imag_error(trunk_epoch_train_error_imag.item())
        trunk_evaluator.store_epoch_learning_rate(epoch_learning_rate)

    # -------------------------------------------------- Trunk Decomposition ---------------------------------------------------------------------------
    
    with torch.no_grad():
        phi = model.trunk_network(trunk_training_data['xt'])
        if p['TRUNK_DECOMPOSITION'].lower() == 'qr':
            Q, R = torch.linalg.qr(phi)
            
        if p['TRUNK_DECOMPOSITION'].lower() == 'svd':
            Q, Sd, Vd = torch.linalg.svd(phi, full_matrices=False)
            R = torch.diag(Sd) @ Vd

        T = torch.linalg.inv(R)
        print(Q.shape, R.shape, T.shape)
        model.set_Q(Q)
        model.set_R(R)
        model.set_T(T)

    # -------------------------------------------------- Branch Training ---------------------------------------------------------------------------
    
    model.set_training_phase('branch')
    model.freeze_trunk()
    model.freeze_A()
    model.unfreeze_branch()

    for epoch in tqdm(range(p['BRANCH_TRAIN_EPOCHS']), colour='GREEN'):
        branch_epoch_train_loss = 0
        branch_epoch_train_error_real = 0
        branch_epoch_train_error_imag = 0
        branch_epoch_learning_rate = branch_scheduler.get_last_lr()[-1]

        branch_train_data = copy.deepcopy(full_batch_train)

        if p['INPUT_NORMALIZATION']:
            branch_train_data = {key: (normalize_branch(value) if key == 'xb' \
                        else value)
                for key, value in branch_train_data.items()}

        branch_train_outputs = branch_trainer(branch_train_data)
        branch_epoch_train_loss = branch_train_outputs['loss']

        if p['OUTPUT_NORMALIZATION']:
            branch_pred_real = denormalize_g_u_real(branch_train_outputs['pred_real'])
            branch_g_u_real = denormalize_g_u_real(branch_train_outputs['coefs_real'])
            branch_pred_imag = denormalize_g_u_imag(branch_train_outputs['pred_imag'])
            branch_g_u_imag = denormalize_g_u_imag(branch_train_outputs['coefs_imag'])

        else:
            branch_pred_real = branch_train_outputs['pred_real']
            branch_g_u_real = branch_train_outputs['coefs_real']
            branch_pred_imag = branch_train_outputs['pred_imag']
            branch_g_u_imag = branch_train_outputs['coefs_imag']
        
        branch_epoch_train_error_real = branch_evaluator.compute_error(branch_g_u_real,
                                                                        branch_pred_real)
        branch_epoch_train_error_imag = branch_evaluator.compute_error(branch_g_u_imag,
                                                                        branch_pred_imag)

        if p['BRANCH_CHANGE_OPTIMIZER']:
            if epoch == p['BRANCH_CHANGE_AT_EPOCH']:
                branch_trainer.optimizer = torch.optim.Adam(list(model.branch_network.parameters()),
                                                            lr=branch_scheduler.get_last_lr()[-1])
                p['BRANCH_LR_SCHEDULING'] = False

        if p['BRANCH_LR_SCHEDULING']:
            branch_scheduler.step()

        if epoch % 500 == 0:
            print(f"Branch loss for epoch {epoch}: {branch_epoch_train_loss:.3E}")

        branch_evaluator.store_epoch_train_loss(branch_epoch_train_loss)
        branch_evaluator.store_epoch_train_real_error(branch_epoch_train_error_real.item())
        branch_evaluator.store_epoch_train_imag_error(branch_epoch_train_error_imag.item())
        
        branch_evaluator.store_epoch_learning_rate(branch_epoch_learning_rate)
        
    if branch_epoch_train_error_real < best_avg_error_real:
        best_avg_error_real = branch_epoch_train_error_real
        best_model_checkpoint = {
            'model_state_dict': model.state_dict(),
                'Q': model.Q,
                'T': model.T,
                'R': model.R,
            }

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
                batch = {key: (ppr.trunk_feature_expansion(value, p['TRUNK_EXPANSION_FEATURES_NUMBER']) if key == 'xt' else value)
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
        evaluator.store_epoch_train_real_error(avg_epoch_train_error_real.item())
        evaluator.store_epoch_train_imag_error(avg_epoch_train_error_imag.item())
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
                batch = {key: (ppr.trunk_feature_expansion(value, p['TRUNK_EXPANSION_FEATURES_NUMBER']) if key == 'xt' else value)
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

        if epoch % 500 == 0:
            print(f"Loss for epoch {epoch}: {epoch_val_loss:.3E}")

        avg_epoch_val_loss = epoch_val_loss / niter_per_val_epoch
        avg_epoch_val_error_real = epoch_val_error_real / niter_per_val_epoch
        avg_epoch_val_error_imag = epoch_val_error_imag / niter_per_val_epoch

        evaluator.store_epoch_val_loss(avg_epoch_val_loss)
        evaluator.store_epoch_val_real_error(avg_epoch_val_error_real)
        evaluator.store_epoch_val_imag_error(avg_epoch_val_error_imag)

        if avg_epoch_val_error_real < best_avg_error_real:
            best_avg_error_real = avg_epoch_val_error_real
            best_model_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
    
    last_model = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
        }
    
end_time = time.time()

# -------------------------------  Getting info from training ---------------------

if p["TWO_STEP_TRAINING"]:
    trunk_loss_history = trunk_evaluator.get_loss_history()
    trunk_error_history = trunk_evaluator.get_error_history()
    trunk_lr_history = trunk_evaluator.get_lr_history()
    trunk_history = {'loss' : trunk_loss_history,
            'error' : trunk_error_history,
            'learning_rate': trunk_lr_history,
            } 
    branch_loss_history = branch_evaluator.get_loss_history()
    branch_error_history = branch_evaluator.get_error_history()
    branch_lr_history = branch_evaluator.get_lr_history()
    branch_history = {'loss' : branch_loss_history,
            'error' : branch_error_history,
            'learning_rate': branch_lr_history,
            }
else:
    loss_history = evaluator.get_loss_history()
    error_history = evaluator.get_error_history()
    lr_history = evaluator.get_lr_history()

    history = {'loss' : loss_history,
            'error' : error_history,
            'learning_rate': lr_history,
            }

training_time = {'time': end_time - start_time}
print(f"Training concluded in: {end_time - start_time} s")

# ------------------------------------ Plot --------------------------------
if  p["TWO_STEP_TRAINING"]:
    trunk_epochs_plot = [i for i in range(p['TRUNK_TRAIN_EPOCHS'])]
    trunk_fig = plot_training(trunk_epochs_plot, trunk_history)
    branch_epochs_plot = [i for i in range(p['BRANCH_TRAIN_EPOCHS'])]
    branch_fig = plot_training(branch_epochs_plot, branch_history)
elif p['PROPER_ORTHOGONAL_DECOMPOSITION']:
    epochs_plot = [i for i in range(epochs)]
    fig = plot_training(epochs_plot, history)
else:
    epochs_plot = [i for i in range(epochs)]
    fig = plot_training(epochs_plot, history)

# --------------------------- Save output -------------------------------
if p["TWO_STEP_TRAINING"]:
    saver(model_state=best_model_checkpoint,
          model_info=p,
          split_indices=dataset_indices,
          norm_params=norm_params,
          time=training_time,
          time_prefix ="training")
    
    saver(history=trunk_history,
          figure=trunk_fig,
          figure_prefix="trunk_history",
          history_prefix='trunk')
    
    saver(history=branch_history,
          figure=branch_fig,
          figure_prefix="branch_history",
          history_prefix='branch')
    
elif p["PROPER_ORTHOGONAL_DECOMPOSITION"]:
    saver(model_state=best_model_checkpoint,
          train_state=last_model,
          model_info=p,
          split_indices=dataset_indices,
          norm_params=norm_params,
          history=history,
          figure=fig,
          time=training_time,
          figure_prefix="history",
          time_prefix ="training")
else:
    saver(model_state=best_model_checkpoint['model_state_dict'],
          train_state=last_model,
          model_info=p,
          split_indices=dataset_indices,
          norm_params=norm_params,
          history=history,
          figure=fig,
          time=training_time,
          figure_prefix="history",
          time_prefix ="training")