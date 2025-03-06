import time
import torch
import logging
from tqdm.auto import tqdm
from ..pipe.saving import Saver
from .store_outputs import HistoryStorer
from ..deeponet.deeponet import DeepONet
from ..data_processing import batching as bt
from ..plotting.plot_training import plot_training, align_epochs
from ..deeponet.training_strategies.helpers import OptimizerSchedulerManager
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
        self.params = params
        if hasattr(training_strategy, "get_phases"):
            self.phases = training_strategy.get_phases()
        else:
            self.phases = [params["TRAINING_STRATEGY"].capitalize()]
        self.storer = HistoryStorer(self.phases)
        self.saver = saver
        self.training_strategy.prepare_training(self.model, params=self.params)
        self.optimizer_manager = OptimizerSchedulerManager(self.params, self.model)
            
    def train(self, train_batch: dict[torch.Tensor], val_batch: dict[torch.Tensor] | None = None) -> dict:
        """
        Executes the training loop. The configuration (via params) defines the pipeline:
         - TRAINING_PHASES: list of phase names (e.g., ["trunk", "branch"] for two-step; ["final"] for single-phase).
        
        The loop iterates over phases, and for each phase:
         - Notifies the training strategy of the current phase.
         - Iterates for the given number of epochs, calling:
           * The model’s forward pass.
           * The training strategy’s loss and error computations.
           * Optimizer steps and scheduler updates.
         - Logs metrics via HistoryStorer.
        """

        if isinstance(self.training_strategy, TwoStepTrainingStrategy):
            epochs_list = [self.params["TRUNK_TRAIN_EPOCHS"], self.params["BRANCH_TRAIN_EPOCHS"]]
        else:
            epochs_list = [self.params["EPOCHS"]]
        
        if len(self.phases) != len(epochs_list):
            raise ValueError("List of epochs don't match number of training phases.")

        best_model_checkpoint = None

        phase_count = -1

        for phase_idx, phase in enumerate(self.phases):
            phase_epochs = epochs_list[phase_idx]
            logger.info(f"Starting phase '{phase}' for {phase_epochs} epochs.")
            
            self.training_strategy.update_training_phase(phase)
            self.training_strategy.prepare_for_phase(self.model, 
                                                     model_params=self.params, 
                                                     train_batch=bt.prepare_batch(train_batch, self.params)
                                                     )
            
            best_train_loss = float('inf')
            best_val_loss = float('inf')

            phase_start = time.time()
            for epoch in tqdm(range(phase_epochs), desc=f"Phase: {phase}", colour=self.params.get("STANDARD_PROGRESS_BAR_COLOR", 'blue')):
                active_optimizer = self.optimizer_manager.get_active_optimizer(epoch)["active"]
                active_scheduler = self.optimizer_manager.get_active_scheduler(epoch)["active"]

                batch = bt.prepare_batch(train_batch, self.params)
                outputs = self.model(batch["xb"], batch["xt"])

                loss = self.training_strategy.compute_loss(outputs, batch, self.model, self.params)

                if epoch % 10 == 0 and epoch > 0:
                    logger.info(f"Loss: {loss:.2E}")

                active_optimizer.zero_grad()
                loss.backward()
                active_optimizer.step()
                if active_scheduler is not None:
                    self.optimizer_manager.step_scheduler(active_scheduler)
                
                errors = self.training_strategy.compute_errors(outputs, batch, self.model, self.params)
                self.storer.store_epoch_train_loss(phase, loss.item())
                self.storer.store_epoch_train_errors(phase, errors)
                self.storer.store_learning_rate(phase, active_optimizer.param_groups[-1]["lr"])

                self._validate(val_batch, phase)

                if loss.item() < best_train_loss:
                    best_train_loss = loss
                    best_model_checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': active_optimizer.state_dict(),
                        'train_loss': best_train_loss,
                        'train_errors': errors
                    }
                    if isinstance(self.training_strategy, TwoStepTrainingStrategy):
                        if phase == 'branch':
                            best_model_checkpoint['trained_trunk'] = self.model.trunk.trained_tensor
                    if isinstance(self.training_strategy, PODTrainingStrategy):
                        best_model_checkpoint['pod_basis'] = self.model.trunk.basis
                        best_model_checkpoint['mean_functions'] = self.model.trunk.mean

                self.training_strategy.after_epoch(epoch, self.model, self.params, train_batch=batch["xt"])

            phase_duration = time.time() - phase_start
            logger.info(f"Phase '{phase}' completed in {phase_duration:.2f} seconds.")
        
        phase_count += 1

        trained_model_config = self._finalize_training(phase_count, best_model_checkpoint, phase_duration)

        return trained_model_config

    def _validate(self, val_batch: dict[str, torch.Tensor], phase: str) -> dict[float, float] | None:
        if isinstance(self.training_strategy, TwoStepTrainingStrategy):
            return
        self.model.eval()
        val_batch_processed = bt.prepare_batch(val_batch, self.params)
        with torch.no_grad():
            val_outputs = self.model(val_batch_processed['xb'], val_batch_processed['xt'])
            val_loss = self.training_strategy.compute_loss(val_outputs, val_batch_processed, self.model, self.params)
            val_errors = self.training_strategy.compute_errors(val_outputs, val_batch_processed, self.model, self.params)

        self.storer.store_epoch_val_loss(phase, val_loss.item())
        self.storer.store_epoch_val_errors(phase, val_errors)

    def _log_epoch_metrics(self, epoch: int, train_loss: float, train_errors: dict[str, list[float]], val_metrics: dict[str, list[float]]) -> None:
        output_errors_str = ", ".join([f"{key}: {train_errors.get(key, 0):.3E}" for key in self.params['OUTPUT_KEYS']])
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.3E}, Train Errors: {output_errors_str}"
        if val_metrics:
            val_output_errors_str = ", ".join(
                [f"{key}: {val_metrics.get('val_error_' + key, 0):.3E}" for key in self.params['OUTPUT_KEYS']]
            )
            log_msg += f", Val Loss: {val_metrics['val_loss']:.3E}, Val Errors: {val_output_errors_str}"
        logger.info(log_msg)

    def _finalize_training(self, phase_count: int, best_model_checkpoint: dict, training_time: float) -> dict[str, any]:
        """
        Finalizes training for the current phase, saving metrics and generating plots.
        """
        history = self.storer.get_history()

        valid_history = {phase: metrics for phase, metrics in history.items() if metrics.get('train_loss')}


        if not valid_history:
            raise ValueError("There's no history to save. There's an error somewhere when generating the history.")

        aligned_history = align_epochs(valid_history)
        fig = plot_training(aligned_history)

        self.saver(
            phase=self.phases[phase_count],
            model_state=best_model_checkpoint,
            model_info=self.params,
            split_indices=self.params['TRAIN_INDICES'],
            norm_params=self.params['NORMALIZATION_PARAMETERS'],
            figure=fig,
            history=valid_history,
            time=training_time,
            figure_prefix=f"{self.phases[phase_count]}",
            history_prefix=f"{self.phases[phase_count]}",
            time_prefix=f"{self.phases[phase_count]}",
        )

        return self.params