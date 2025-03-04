import time
import torch
import logging
from tqdm.auto import tqdm
from ..pipe.saving import Saver
from .store_ouptuts import HistoryStorer
from . import preprocessing as ppr
from ..deeponet.deeponet import DeepONet
from ..plotting.plot_training import plot_training, align_epochs
from ..deeponet.training_strategies import (
    TrainingStrategy,
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)

logger = logging.getLogger(__name__)

class TrainingLoop:
    def __init__(self, model: DeepONet, training_strategy: TrainingStrategy, saver: Saver, params: dict) -> None:
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
        self.params = params
        self.training_strategy.prepare_training(self.model, params=self.params)
        self.optimizers = self.training_strategy.get_optimizers(self.model, self.params)
        self.schedulers = self.training_strategy.get_schedulers(self.optimizers, self.params)
    
    def train(self, train_batch: dict[torch.Tensor], val_batch: dict[torch.Tensor] | None=None) -> dict:
        epochs_per_phase = self.training_strategy.get_epochs(self.params)
        best_model_checkpoint = None
        best_train_loss = float('inf')
        best_val_loss = float('inf')

        for phase_index, phase_epochs in enumerate(epochs_per_phase):
            phase_start_time = time.time()

            current_phase = self.training_strategy.phases[phase_index]
            train_batch_processed = ppr.prepare_batch(train_batch, self.params)
            self.training_strategy.update_training_phase(current_phase)
            self.training_strategy.prepare_for_phase(self.model, 
                                                    model_params=self.params, 
                                                    train_batch=train_batch_processed['xt'])

            logger.info(f"Starting phase: {current_phase}, Epochs: {phase_epochs}")

            progress_bar_color = self.params[current_phase.upper() + '_' + 'PROGRESS_BAR_COLOR'] if self.params['TRAINING_STRATEGY'] == 'two_step' else \
                              self.params[self.params['TRAINING_STRATEGY'].upper() + '_' + 'PROGRESS_BAR_COLOR']

            for epoch in tqdm(range(phase_epochs), 
                              desc=f"Phase {current_phase}", 
                              colour=progress_bar_color):

                train_batch_processed = ppr.prepare_batch(train_batch, self.params)

                outputs = self.model(train_batch_processed['xb'], train_batch_processed['xt'])
                loss = self.training_strategy.compute_loss(outputs, train_batch_processed, self.model, self.params)

                self.training_strategy.zero_grad(self.optimizers)
                loss.backward()

                if epoch % 500 == 0 and epoch > 0:
                    logger.info(f"\nLoss: {loss.item():.3E}\n")

                self.training_strategy.step(self.optimizers)

                errors = self.training_strategy.compute_errors(outputs, train_batch_processed, self.model, self.params)

                self.storer.store_epoch_train_loss(current_phase, loss.item())
                self.storer.store_epoch_train_errors(current_phase, errors)
                self.storer.store_learning_rate(current_phase, self.optimizers[self.training_strategy.current_phase].param_groups[-1]['lr'])

                if self.training_strategy.can_validate() and val_batch:
                    val_metrics = self._validate(val_batch)
                    val_loss = val_metrics['val_loss']
                    self.storer.store_epoch_val_loss(current_phase, val_metrics['val_loss'])
                    val_errors = {f"{key}": val_metrics.get(f"{key}", None)
                                  for key in self.params['OUTPUT_KEYS']}

                    self.storer.store_epoch_val_errors(current_phase, val_errors)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizers['optimizer'].state_dict() if self.optimizers.get('optimizer') else None,
                            'val_loss': val_loss,
                            'val_errors': val_errors
                        }
                else:
                    if loss < best_train_loss:
                        best_train_loss = loss
                    if isinstance(self.training_strategy, TwoStepTrainingStrategy):
                        best_model_checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'Q': self.training_strategy.Q,
                            'T': self.training_strategy.T,
                            'R': self.training_strategy.R,
                            'optimizer_state_dict': self.optimizers[self.training_strategy.current_phase].state_dict() if self.optimizers.get(self.training_strategy.current_phase) else None,
                            'train_loss': loss,
                            'val_loss': None
                        }
                
                if isinstance(self.training_strategy, PODTrainingStrategy):
                    best_model_checkpoint['pod_basis'] = self.training_strategy.pod_basis
                    best_model_checkpoint['mean_functions'] = self.training_strategy.mean_functions
                
                if epoch < self.params[self.training_strategy.current_phase.upper() + '_CHANGE_AT_EPOCH']:
                    self.training_strategy.step_schedulers(self.schedulers)
                self.training_strategy.after_epoch(epoch, self.model, self.params, train_batch=train_batch_processed['xt'])

            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time

            trained_model_info = self._finalize_training(best_model_checkpoint, training_time=phase_duration)

        return trained_model_info

    def _validate(self, val_batch: dict[str, torch.Tensor]) -> dict[float, float]:
        self.model.eval()
        val_batch_processed = ppr.prepare_batch(val_batch, self.params)
        with torch.no_grad():
            val_outputs = self.model(val_batch_processed['xb'], val_batch_processed['xt'])
            val_loss = self.training_strategy.compute_loss(val_outputs, val_batch_processed, self.model, self.params)
            val_errors = self.training_strategy.compute_errors(val_outputs, val_batch_processed, self.model, self.params)

        val_metrics = {'val_loss': val_loss.item()}
        for key in self.params['OUTPUT_KEYS']:
            val_metrics[f"{key}"] = val_errors.get(key, None)
        return val_metrics

    def _log_epoch_metrics(self, epoch: int, train_loss: float, train_errors: dict[str, list[float]], val_metrics: dict[str, list[float]]) -> None:
        output_errors_str = ", ".join([f"{key}: {train_errors.get(key, 0):.3E}" for key in self.params['OUTPUT_KEYS']])
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.3E}, Train Errors: {output_errors_str}"
        if val_metrics:
            val_output_errors_str = ", ".join(
                [f"{key}: {val_metrics.get('val_error_' + key, 0):.3E}" for key in self.params['OUTPUT_KEYS']]
            )
            log_msg += f", Val Loss: {val_metrics['val_loss']:.3E}, Val Errors: {val_output_errors_str}"
        logger.info(log_msg)

    def _finalize_training(self, best_model_checkpoint: dict, training_time: float) -> dict[str, any]:
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
            model_info=self.params,
            split_indices=self.params['TRAIN_INDICES'],
            norm_params=self.params['NORMALIZATION_PARAMETERS'],
            figure=fig,
            history=valid_history,
            time=training_time,
            figure_prefix=f"phase_{self.training_strategy.current_phase}",
            history_prefix=f"phase_{self.training_strategy.current_phase}",
            time_prefix=f"phase_{self.training_strategy.current_phase}",
        )

        return self.params