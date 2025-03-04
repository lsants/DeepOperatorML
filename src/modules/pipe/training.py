import time
import torch
import logging
from typing import Dict, Any, Optional
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
        self.optimizers = self.training_strategy.get_optimizers(self.model, self.params) # Integrate with optimizer factory
        self.schedulers = self.training_strategy.get_schedulers(self.optimizers, self.params)
            
    def train(self, train_batch: dict[torch.Tensor], val_batch: dict[torch.Tensor] | None = None) -> dict:
        """
        Executes the training loop. The configuration (via params) defines the pipeline:
         - TRAINING_PHASES: list of phase names (e.g., ["trunk", "branch"] for two-step; ["default"] for single-phase).
         - EPOCHS_PER_PHASE: corresponding list of epoch counts.
        
        The loop iterates over phases, and for each phase:
         - Notifies the training strategy of the current phase.
         - Iterates for the given number of epochs, calling:
           * The model’s forward pass.
           * The training strategy’s loss and error computations.
           * Optimizer steps and scheduler updates.
         - Logs metrics via HistoryStorer.
        """
        phases = self.params.get("TRAINING_PHASES", ["default"])
        epochs_list = self.params.get("EPOCHS_PER_PHASE", [self.params.get("EPOCHS", 1000)])
        if len(phases) != len(epochs_list):
            raise ValueError("TRAINING_PHASES and EPOCHS_PER_PHASE lengths do not match.")

        best_model_checkpoint = None
        best_loss = float('inf')

        for phase_idx, phase in enumerate(phases):
            phase_epochs = epochs_list[phase_idx]
            logger.info(f"Starting phase '{phase}' for {phase_epochs} epochs.")
            
            # Notify training strategy about the current phase.
            self.training_strategy.update_training_phase(phase)
            self.training_strategy.prepare_for_phase(self.model, model_params=self.params, train_batch=train_batch)

            phase_start = time.time()
            for epoch in tqdm(range(phase_epochs), desc=f"Phase: {phase}", colour="green"):
                batch = ppr.prepare_batch(train_batch, self.params)
                outputs = self.model(batch["xb"], batch["xt"])
                loss = self.training_strategy.compute_loss(outputs, batch, self.model, self.params)
                self.training_strategy.zero_grad(self.optimizers[phase])
                loss.backward()
                self.training_strategy.step(self.optimizers[phase])
                # Update schedulers for the current phase.
                self.training_strategy.step_scheduler(self.schedulers[phase])
                
                errors = self.training_strategy.compute_errors(outputs, batch, self.model, self.params)
                self.storer.store(phase, epoch, loss.item(), errors,
                                  lr=self.optimizers[phase].param_groups[-1]["lr"])
                self.training_strategy.after_epoch(epoch, self.model, self.params, train_batch=batch["xt"])

            phase_duration = time.time() - phase_start
            logger.info(f"Phase '{phase}' completed in {phase_duration:.2f} seconds.")
            # Optionally: checkpoint saving, plotting, etc.
            best_model_checkpoint = self.saver.save(phase, self.model, self.params, phase_duration)

        return {"best_model_checkpoint": best_model_checkpoint}                

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