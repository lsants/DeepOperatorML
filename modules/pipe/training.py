import time
import torch
from tqdm.auto import tqdm
from .store_ouptuts import HistoryStorer
from ..data_processing import preprocessing as ppr
from ..plotting.plot_training import plot_training, align_epochs

class TrainingLoop:
    def __init__(self, model, training_strategy, saver, params):
        """
        Initializes the TrainingLoop.

        Args:
            model (torch.nn.Module): The model to train.
            training_strategy (TrainingStrategy): The training strategy to use.
            storer (TrainEvaluator): The storer for metrics.
            saver (Saver): The saver for saving models and outputs.
            params (dict): Training parameters.
        """
        self.model = model
        self.training_strategy = training_strategy
        self.storer = HistoryStorer(training_strategy.get_phases()) 
        self.saver = saver
        self.p = params

        self.training_strategy.prepare_training(self.model)
        self.optimizers = self.training_strategy.get_optimizers(self.model, self.p)
        self.schedulers = self.training_strategy.get_schedulers(self.optimizers, self.p)

    def prepare_batch(self, batch):
        """
        Prepares the batch data, including normalization and feature expansion.

        Args:
            batch (dict): The batch data.

        Returns:
            dict: The processed batch data.
        """
        processed_batch = {}
        dtype = getattr(torch, self.p['PRECISION'])
        device = self.p['DEVICE']

        xb_scaler = ppr.Scaling(
            min_val=self.p['NORMALIZATION_PARAMETERS']['xb']['min'],
            max_val=self.p['NORMALIZATION_PARAMETERS']['xb']['max']
        )
        xt_scaler = ppr.Scaling(
            min_val=self.p['NORMALIZATION_PARAMETERS']['xt']['min'],
            max_val=self.p['NORMALIZATION_PARAMETERS']['xt']['max']
        )
        g_u_real_scaler = ppr.Scaling(
            min_val=self.p['NORMALIZATION_PARAMETERS']['g_u_real']['min'],
            max_val=self.p['NORMALIZATION_PARAMETERS']['g_u_real']['max']
        )
        g_u_imag_scaler = ppr.Scaling(
            min_val=self.p['NORMALIZATION_PARAMETERS']['g_u_imag']['min'],
            max_val=self.p['NORMALIZATION_PARAMETERS']['g_u_imag']['max']
        )

        if self.p['INPUT_NORMALIZATION']:
            processed_batch['xb'] = xb_scaler.normalize(batch['xb']).to(dtype=dtype, device=device)
            processed_batch['xt'] = xt_scaler.normalize(batch['xt']).to(dtype=dtype, device=device)
        else:
            processed_batch['xb'] = batch['xb'].to(dtype=dtype, device=device)
            processed_batch['xt'] = batch['xt'].to(dtype=dtype, device=device)

        if self.p['OUTPUT_NORMALIZATION']:
            processed_batch['g_u_real'] = g_u_real_scaler.normalize(batch['g_u_real']).to(dtype=dtype, device=device)
            processed_batch['g_u_imag'] = g_u_imag_scaler.normalize(batch['g_u_imag']).to(dtype=dtype, device=device)
        else:
            processed_batch['g_u_real'] = batch['g_u_real'].to(dtype=dtype, device=device)
            processed_batch['g_u_imag'] = batch['g_u_imag'].to(dtype=dtype, device=device)

        if self.p['TRUNK_FEATURE_EXPANSION']:
            processed_batch['xt'] = ppr.trunk_feature_expansion(
                processed_batch['xt'], self.p['TRUNK_EXPANSION_FEATURES_NUMBER']
            )

        return processed_batch
    
    def train(self, train_batch, val_batch=None):
        epochs_per_phase = self.training_strategy.get_epochs(self.p)
        best_model_checkpoint = None
        best_val_loss = float('inf')

        for phase_index, phase_epochs in enumerate(epochs_per_phase):
            phase_start_time = time.time()

            current_phase = self.training_strategy.phases[phase_index]
            train_batch_processed = self.prepare_batch(train_batch)
            self.training_strategy.update_training_phase(current_phase)
            self.training_strategy.prepare_for_phase(self.model, 
                                                    model_params=self.p, 
                                                    train_batch=train_batch_processed['xt'])
 

            print(f"Starting phase: {current_phase}, Epochs: {phase_epochs}")

            for epoch in tqdm(range(phase_epochs), desc=f"Phase {current_phase}", colour=self.p['PROGRESS_BAR_COLOR']):

                train_batch_processed = self.prepare_batch(train_batch)

                outputs = self.model(train_batch_processed['xb'], train_batch_processed['xt'])

                loss = self.training_strategy.compute_loss(outputs, train_batch_processed, self.model, self.p)

                self.training_strategy.zero_grad(self.optimizers)
                loss.backward()
                
                # print("==== Gradients of Parameters ====")
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter: {name}")
                #         if param.grad is not None:
                #             print(f"Gradient Mean: {param.grad.mean().item()}, Std: {param.grad.std().item()}")
                #         else:
                #             print("No gradient computed for this parameter!")
                            
               
                # if hasattr(self.training_strategy, "A_list"):
                #     for i, A in enumerate(self.training_strategy.A_list):
                #         print(f"A_list[{i}] gradient:")
                #         if A.grad is not None:
                #             print(f"Gradient Mean: {A.grad.mean().item()}, Std: {A.grad.std().item()}")
                #         else:
                #             print("No gradient computed for A_list!")
                # print("==============================")

                print(f"Loss: {loss.item()}")

                # print(self.optimizers)

                self.training_strategy.step(self.optimizers)


                # Print parameter names and shapes for the optimizer in each phase



                # print(f"Optimizer Params for Branch Phase: {[p.size() for p in self.optimizers['optimizer'].param_groups[0]['params']]}")




                errors = self.training_strategy.compute_errors(outputs, train_batch_processed, self.model, self.p)

                # for group in optimizer.param_groups:
                #     for param in group['params']:
                #         print(f"Optimizer param: {param.shape}, requires_grad={param.requires_grad}")



                self.storer.store_epoch_train_loss(current_phase, loss.item())
                self.storer.store_epoch_train_errors(current_phase, errors)

                if self.training_strategy.can_validate() and val_batch:
                    val_metrics = self._validate(val_batch)
                    val_loss = val_metrics['val_loss']
                    self.storer.store_epoch_val_loss(current_phase, val_metrics['val_loss'])
                    self.storer.store_epoch_val_errors(current_phase, {
                        'real': val_metrics.get('val_error_real', None),
                        'imag': val_metrics.get('val_error_imag', None),
                    })
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizers['optimizer'].state_dict() if self.optimizers.get('optimizer') else None,
                            'val_loss': val_loss
                        }

                if not self.training_strategy.can_validate():
                    best_model_checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'Q': self.training_strategy.Q_list,
                        'T': self.training_strategy.T_list,
                        'R': self.training_strategy.R_list,
                        'optimizer_state_dict': self.optimizers['optimizer'].state_dict() if self.optimizers.get('optimizer') else None,
                        'val_loss': None
                    }
                self.training_strategy.step_schedulers(self.schedulers)
                self.training_strategy.after_epoch(epoch, self.model, self.p, train_batch=train_batch_processed['xt'])
            
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time

            self._finalize_training(best_model_checkpoint, training_time=phase_duration)

    def _validate(self, val_batch):
        self.model.eval()
        val_batch_processed = self.prepare_batch(val_batch)
        with torch.no_grad():
            val_outputs = self.model(val_batch_processed['xb'], val_batch_processed['xt'])
            val_loss = self.training_strategy.compute_loss(val_outputs, val_batch_processed, self.model, self.p)
            val_errors = self.training_strategy.compute_errors(val_outputs, val_batch_processed, self.model, self.p)

        return {
            'val_loss': val_loss.item(),
            'val_error_real': val_errors.get('g_u_real', None),
            'val_error_imag': val_errors.get('g_u_imag', None),
        }

    def _log_epoch_metrics(self, epoch, train_loss, train_errors, val_metrics):
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.3E}, Train Error Real: {train_errors['g_u_real']:.3E}, Train Error Imag: {train_errors['g_u_imag']:.3E}"
        if val_metrics:
            log_msg += f", Val Loss: {val_metrics['val_loss']:.3E}, Val Error Real: {val_metrics['val_error_real']:.3E}"
        print(log_msg)

    def _finalize_training(self, best_model_checkpoint, training_time):
        """
        Finalizes training for the current phase, saving metrics and generating plots.
        """
        history = self.storer.get_history()

        valid_history = {phase: metrics for phase, metrics in history.items() if metrics.get('train_loss')}

        if not valid_history:
            print("No valid phases to save. Skipping finalization.")
            return

        aligned_history = align_epochs(valid_history)  # Align data before plotting
        fig = plot_training(aligned_history)  # Pass aligned data

        self.saver(
            phase=self.training_strategy.current_phase,
            model_state=best_model_checkpoint,
            model_info=self.p,
            split_indices=self.p['TRAIN_INDICES'],
            norm_params=self.p['NORMALIZATION_PARAMETERS'],
            figure=fig,
            history=valid_history,
            time=training_time,
            figure_prefix=f"phase_{self.training_strategy.current_phase}",
            history_prefix=f"phase_{self.training_strategy.current_phase}",
            time_prefix=f"phase_{self.training_strategy.current_phase}",
        )



