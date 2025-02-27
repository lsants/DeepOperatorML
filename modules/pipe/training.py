import time
import torch
import logging
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)
from .store_ouptuts import HistoryStorer
from ..data_processing import preprocessing as ppr
from ..plotting.plot_training import plot_training, align_epochs
from ..deeponet.training_strategies import (
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)

class TrainingLoop:
    def __init__(self, model, training_strategy, saver, params):
        """
        Initializes the TrainingLoop.

        Args:
            model (torch.nn.Module): The model to train.
            training_strategy (TrainingStrategy): The training strategy to use.
            saver (Saver): The saver for saving models and outputs.
            params (dict): Training parameters.
        """
        self.model = model
        self.training_strategy = training_strategy
        self.storer = HistoryStorer(training_strategy.get_phases()) 
        self.saver = saver
        self.p = params

        self.training_strategy.prepare_training(self.model, params=self.p)
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

        if self.p['INPUT_NORMALIZATION']:
            processed_batch['xb'] = xb_scaler.normalize(batch['xb']).to(dtype=dtype, device=device)
            processed_batch['xt'] = xt_scaler.normalize(batch['xt']).to(dtype=dtype, device=device)
        else:
            processed_batch['xb'] = batch['xb'].to(dtype=dtype, device=device)
            processed_batch['xt'] = batch['xt'].to(dtype=dtype, device=device)

        for key in self.p['OUTPUT_KEYS']:
            scaler = ppr.Scaling(
                min_val=self.p['NORMALIZATION_PARAMETERS'][key]['min'],
                max_val=self.p['NORMALIZATION_PARAMETERS'][key]['max']
            )
            if self.p['OUTPUT_NORMALIZATION']:
                processed_batch[key] = scaler.normalize(batch[key]).to(dtype=dtype, device=device)
            else:
                processed_batch[key] = batch[key].to(dtype=dtype, device=device)

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

            logger.info(f"Starting phase: {current_phase}, Epochs: {phase_epochs}")

            progress_bar_color = self.p[current_phase.upper() + '_' + 'PROGRESS_BAR_COLOR'] if self.p['TRAINING_STRATEGY'] == 'two_step' else \
                              self.p[self.p['TRAINING_STRATEGY'].upper() + '_' + 'PROGRESS_BAR_COLOR']


            for epoch in tqdm(range(phase_epochs), 
                              desc=f"Phase {current_phase}", 
                              colour=progress_bar_color):

                train_batch_processed = self.prepare_batch(train_batch)

                outputs = self.model(train_batch_processed['xb'], train_batch_processed['xt'])
                loss = self.training_strategy.compute_loss(outputs, train_batch_processed, self.model, self.p)

                self.training_strategy.zero_grad(self.optimizers)
                loss.backward()

                # if epoch % 500 == 0:
                #     logger.info(f"\nLoss: {loss.item():.3E}\n")

                self.training_strategy.step(self.optimizers)

                errors = self.training_strategy.compute_errors(outputs, train_batch_processed, self.model, self.p)

                self.storer.store_epoch_train_loss(current_phase, loss.item())
                self.storer.store_epoch_train_errors(current_phase, errors)
                self.storer.store_learning_rate(current_phase, self.optimizers[self.training_strategy.current_phase].param_groups[-1]['lr'])

                if self.training_strategy.can_validate() and val_batch:
                    val_metrics = self._validate(val_batch)
                    val_loss = val_metrics['val_loss']
                    self.storer.store_epoch_val_loss(current_phase, val_metrics['val_loss'])
                    val_errors = {f"{key}": val_metrics.get(f"{key}", None)
                                  for key in self.p['OUTPUT_KEYS']}

                    self.storer.store_epoch_val_errors(current_phase, val_errors)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizers['optimizer'].state_dict() if self.optimizers.get('optimizer') else None,
                            'val_loss': val_loss,
                            'val_errors': val_errors
                        }

                if not self.training_strategy.can_validate():
                    if isinstance(self.training_strategy, TwoStepTrainingStrategy):
                        best_model_checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'Q': self.training_strategy.Q_list,
                            'T': self.training_strategy.T_list,
                            'R': self.training_strategy.R_list,
                            'optimizer_state_dict': self.optimizers[self.training_strategy.current_phase].state_dict() if self.optimizers.get(self.training_strategy.current_phase) else None,
                            'val_loss': None
                        }
                
                if isinstance(self.training_strategy, PODTrainingStrategy):
                    best_model_checkpoint['pod_basis'] = self.training_strategy.pod_basis
                    best_model_checkpoint['mean_functions'] = self.training_strategy.mean_functions
                
                if epoch < self.p[self.training_strategy.current_phase.upper() + '_CHANGE_AT_EPOCH']:
                    self.training_strategy.step_schedulers(self.schedulers)
                self.training_strategy.after_epoch(epoch, self.model, self.p, train_batch=train_batch_processed['xt'])

            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time

            trained_model_info = self._finalize_training(best_model_checkpoint, training_time=phase_duration)
        return trained_model_info

    def _validate(self, val_batch):
        self.model.eval()
        val_batch_processed = self.prepare_batch(val_batch)
        with torch.no_grad():
            val_outputs = self.model(val_batch_processed['xb'], val_batch_processed['xt'])
            val_loss = self.training_strategy.compute_loss(val_outputs, val_batch_processed, self.model, self.p)
            val_errors = self.training_strategy.compute_errors(val_outputs, val_batch_processed, self.model, self.p)

        val_metrics = {'val_loss': val_loss.item()}
        for key in self.p['OUTPUT_KEYS']:
            val_metrics[f"{key}"] = val_errors.get(key, None)
        return val_metrics

    def _log_epoch_metrics(self, epoch, train_loss, train_errors, val_metrics):
        output_errors_str = ", ".join([f"{key}: {train_errors.get(key, 0):.3E}" for key in self.p['OUTPUT_KEYS']])
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.3E}, Train Errors: {output_errors_str}"
        if val_metrics:
            val_output_errors_str = ", ".join(
                [f"{key}: {val_metrics.get('val_error_' + key, 0):.3E}" for key in self.p['OUTPUT_KEYS']]
            )
            log_msg += f", Val Loss: {val_metrics['val_loss']:.3E}, Val Errors: {val_output_errors_str}"
        logger.info(log_msg)

    def _finalize_training(self, best_model_checkpoint, training_time):
        """
        Finalizes training for the current phase, saving metrics and generating plots.
        """
        history = self.storer.get_history()

        valid_history = {phase: metrics for phase, metrics in history.items() if metrics.get('train_loss')}

        if not valid_history:
            logger.info("No valid phases to save. Skipping finalization.")
            return

        aligned_history = align_epochs(valid_history)
        fig = plot_training(aligned_history)

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

        return self.p



