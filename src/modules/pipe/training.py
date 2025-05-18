from __future__ import annotations
import os
import time
import torch
import logging
from tqdm.auto import tqdm
from typing import Optional, Any
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
    def __init__(self, model: DeepONet, training_params: dict) -> None:
        """
        Initializes the TrainingLoop.

        Args:
            model (torch.nn.Module): The model to train.
            training_strategy (TrainingStrategy): The training strategy to use.
            saver (Saver): The saver for saving models and outputs.
            training_params (dict): Training parameters.
        """
        self.model = model
        self.training_strategy = model.training_strategy
        self.training_params = training_params
        if hasattr(self.training_strategy, "get_phases"):
            self.phases = self.training_strategy.get_phases()  # type: ignore
        else:
            self.phases = [training_params["TRAINING_STRATEGY"].capitalize()]
        self.storer = HistoryStorer(phases=self.phases)
        self.training_strategy.prepare_for_training(
            model=self.model, training_params=self.training_params)
        self.optimizer_manager = OptimizerSchedulerManager(
            model_config=self.training_params, model=self.model)

    def train(self, train_batch: dict[str, torch.Tensor], val_batch: dict[str, torch.Tensor] | None = None) -> dict:
        """
        Executes the training loop. The configuration (via training_params) defines the pipeline:
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
            epochs_list = [self.training_params["TRUNK_TRAIN_EPOCHS"],
                           self.training_params["BRANCH_TRAIN_EPOCHS"]]
        else:
            epochs_list = [self.training_params["EPOCHS"]]

        if len(self.phases) != len(epochs_list):
            raise ValueError(
                "List of epochs don't match number of training phases.")

        best_model_checkpoint = None

        phase_count = -1

        for phase_idx, phase in enumerate(self.phases):
            phase_epochs = epochs_list[phase_idx]
            logger.info(f"Starting phase '{phase}' for {phase_epochs} epochs.")

            self.training_strategy.update_training_phase(phase)
            self.training_strategy.prepare_for_phase(self.model,
                                                     training_params=self.training_params,
                                                     train_batch=bt.prepare_batch(
                                                         train_batch,
                                                         self.training_params)
                                                     )

            best_train_loss = float('inf')

            phase_start = time.time()
            for epoch in tqdm(range(phase_epochs), desc=f"Phase: {phase}", colour=self.training_params.get("STANDARD_PROGRESS_BAR_COLOR", 'blue')):
                active_optimizer = self.optimizer_manager.get_active_optimizer(epoch, phase)[
                    "active"]
                active_scheduler = self.optimizer_manager.get_active_scheduler(epoch, phase)[
                    "active"]

                batch = bt.prepare_batch(train_batch, self.training_params)
                outputs = self.model(batch["xb"], batch["xt"])

                loss = self.training_strategy.compute_loss(
                    outputs, batch, self.model, training_params=self.training_params)

                if epoch % 100 == 0 and epoch > 0:
                    print(f"Loss: {loss:.2E}")

                active_optimizer.zero_grad()
                loss.backward()
                active_optimizer.step()
                if active_scheduler is not None:
                    self.optimizer_manager.step_scheduler(active_scheduler)

                errors = self.training_strategy.compute_errors(
                    outputs, batch, self.model, training_params=self.training_params)
                self.storer.store_epoch_train_loss(phase, loss.item())
                self.storer.store_epoch_train_errors(phase, errors)
                self.storer.store_learning_rate(
                    phase, active_optimizer.param_groups[-1]["lr"])

                self._validate(val_batch, phase)

                model_checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': active_optimizer.state_dict(),
                    'train_loss': loss.item(),
                    'train_errors': errors,
                    'epoch': epoch
                }
                if isinstance(self.training_strategy, TwoStepTrainingStrategy):
                    if phase == 'branch':
                        model_checkpoint['trained_trunk'] = self.model.trunk.trained_tensor
                if isinstance(self.training_strategy, PODTrainingStrategy):
                    model_checkpoint['pod_basis'] = self.model.trunk.basis
                    model_checkpoint['mean_functions'] = self.model.trunk.mean

                if loss.item() < best_train_loss:
                    best_train_loss = loss
                    best_model_checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': active_optimizer.state_dict(),
                        'train_loss': best_train_loss,
                        'train_errors': errors,
                        'epoch': epoch
                    }
                    if isinstance(self.training_strategy, TwoStepTrainingStrategy):
                        if phase == 'branch':
                            best_model_checkpoint['trained_trunk'] = self.model.trunk.trained_tensor
                    if isinstance(self.training_strategy, PODTrainingStrategy):
                        best_model_checkpoint['pod_basis'] = self.model.trunk.basis

                        best_model_checkpoint['mean_functions'] = self.model.trunk.mean
                self.training_strategy.after_epoch(
                    epoch, self.model, self.training_params, train_batch=batch["xt"])

            phase_duration = time.time() - phase_start
            logger.info(
                f"Phase '{phase}' completed in {phase_duration:.2f} seconds.")

        phase_count += 1

        trained_model_config = self._finalize_training(
            phase_count, model_checkpoint, best_model_checkpoint, phase_duration)

        return trained_model_config

    def _validate(self, val_batch: dict[str, torch.Tensor], phase: str) -> dict[float, float] | None:
        if isinstance(self.training_strategy, TwoStepTrainingStrategy):
            return
        self.model.eval()
        val_batch_processed = bt.prepare_batch(
            val_batch, training_params=self.training_params)
        with torch.no_grad():
            val_outputs = self.model(
                val_batch_processed['xb'], val_batch_processed['xt'])
            val_loss = self.training_strategy.compute_loss(
                val_outputs, val_batch_processed, self.model, training_params=self.training_params)
            val_errors = self.training_strategy.compute_errors(
                val_outputs, val_batch_processed, self.model, training_params=self.training_params)
            print(val_errors['g_u'])
        self.storer.store_epoch_val_loss(phase, val_loss.item())
        self.storer.store_epoch_val_errors(phase, val_errors)

    def _log_epoch_metrics(self, epoch: int, train_loss: float, train_errors: dict[str, list[float]], val_metrics: dict[str, list[float]]) -> None:
        output_errors_str = ", ".join(
            [f"{key}: {train_errors.get(key, 0):.3E}" for key in self.training_params['OUTPUT_KEYS']])
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.3E}, Train Errors: {output_errors_str}"
        if val_metrics:
            val_output_errors_str = ", ".join(
                [f"{key}: {val_metrics.get('val_error_' + key, 0):.3E}" for key in self.training_params['OUTPUT_KEYS']]
            )
            log_msg += f", Val Loss: {val_metrics['val_loss']:.3E}, Val Errors: {val_output_errors_str}"
        logger.info(log_msg)

    def _finalize_training(self, phase_count: int, final_model_checkpoint: dict, best_model_checkpoint: dict, training_time: float) -> dict[str, any]:
        """
        Finalizes training for the current phase, saving metrics and generating plots.
        """
        history = self.storer.get_history()

        valid_history = {phase: metrics for phase,
                         metrics in history.items() if metrics.get('train_loss')}

        if not valid_history:
            raise ValueError(
                "There's no history to save. There's an error somewhere when generating the history.")

        aligned_history = align_epochs(history=valid_history)
        fig = plot_training(history=aligned_history)

        if isinstance(self.training_strategy, TwoStepTrainingStrategy):
            phase = self.phases[phase_count]
        else:
            phase = ''

        best_epoch = best_model_checkpoint['epoch']
        final_epoch = final_model_checkpoint['epoch']

        final_checkpoint_path = os.path.join(
            self.training_params["CHECKPOINTS_PATH"],
            f'epoch_{final_epoch}.pth'
        )
        best_checkpoint_path = os.path.join(
            self.training_params["CHECKPOINTS_PATH"],
            f'epoch_{best_epoch}.pth'
        )
        best_model_path = os.path.join(self.training_params["CHECKPOINTS_PATH"],
                                       f'best_model_state.pth'
                                       )
        fig_path = os.path.join(self.training_params["PLOTS_PATH"],
                                f'training_{phase}_plot.png'
                                )
        history_path = os.path.join(self.training_params["METRICS_PATH"],
                                    f'training_{phase}_history.txt'
                                    )
        time_path = os.path.join(self.training_params["METRICS_PATH"],
                                 f'training_{phase}_time.txt'
                                 )

        self.saver.save_checkpoint(
            file_path=final_checkpoint_path,
            model_dict=final_model_checkpoint
        )
        self.saver.save_checkpoint(
            file_path=best_checkpoint_path,
            model_dict=best_model_checkpoint
        )

        self.saver.save_model_state(
            file_path=best_model_path,
            model_state=best_model_checkpoint['model_state_dict']
        )

        self.saver.save_model_info(
            file_path=self.training_params['MODEL_INFO_PATH'],
            model_info=self.training_params
        )

        self.saver.save_indices(
            file_path=self.training_params['DATASET_INDICES_PATH'],
            indices=self.training_params['TRAIN_INDICES']
        )

        self.saver.save_norm_params(
            file_path=self.training_params['NORM_PARAMS_PATH'],
            norm_params=self.training_params['NORMALIZATION_PARAMETERS']
        )

        self.saver.save_plots(
            file_path=fig_path,
            figure=fig
        )

        self.saver.save_history(
            file_path=history_path,
            history=valid_history
        )

        self.saver.save_time(
            file_path=time_path,
            times=training_time
        )

        return self.training_params
