# modules/training_loop.py

import time
import torch
from . import preprocessing as ppr
from .saving import Saver
from tqdm.auto import tqdm
from .plotting import plot_training
from .train_evaluator import TrainEvaluator

class TrainingLoop:
    def __init__(self, model, training_strategy, evaluator, saver, params):
        """
        Initializes the TrainingLoop.

        Args:
            model (torch.nn.Module): The model to train.
            training_strategy (TrainingStrategy): The training strategy to use.
            evaluator (TrainEvaluator): The evaluator for metrics.
            saver (Saver): The saver for saving models and outputs.
            params (dict): Training parameters.
        """
        self.model = model
        self.training_strategy = training_strategy
        self.evaluator = evaluator
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
        """
        Perform the training loop.

        Args:
            train_batch (dict): The training data as a dictionary.
            val_batch (dict, optional): The validation data as a dictionary.
        """
        epochs = self.p['N_EPOCHS']
        best_avg_error_real = float('inf')
        best_model_checkpoint = None

        start_time = time.time()

        for epoch in tqdm(range(epochs), desc="Training", colour='GREEN'):
            self.training_strategy.before_epoch(epoch, self.model, self.p)

            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_error_real = 0.0
            epoch_train_error_imag = 0.0

            train_batch_processed = self.prepare_batch(train_batch)
            outputs = self.model(train_batch_processed['xb'], train_batch_processed['xt'])

            loss = self.training_strategy.compute_loss(outputs, train_batch_processed, self.model)

            self.training_strategy.zero_grad(self.optimizers)
            loss.backward()
            self.training_strategy.step(self.optimizers)

            errors = self.training_strategy.compute_errors(outputs, train_batch_processed, self.model)

            epoch_train_loss += loss.item()
            epoch_train_error_real += errors['real']
            epoch_train_error_imag += errors['imag']

            self.training_strategy.step_schedulers(self.schedulers)

            self.evaluator.store_epoch_train_loss(epoch_train_loss)
            self.evaluator.store_epoch_train_real_error(epoch_train_error_real)
            self.evaluator.store_epoch_train_imag_error(epoch_train_error_imag)

            if val_batch and self.training_strategy.can_validate():
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_error_real = 0.0
                epoch_val_error_imag = 0.0

                val_batch_processed = self.prepare_batch(val_batch)

                with torch.no_grad():
                    val_outputs = self.model(val_batch_processed['xb'], val_batch_processed['xt'])

                    val_loss = self.training_strategy.compute_loss(val_outputs, val_batch_processed, self.model)
                    epoch_val_loss += val_loss.item()

                    val_errors = self.training_strategy.compute_errors(val_outputs, val_batch_processed, self.model)
                    epoch_val_error_real += val_errors['real']
                    epoch_val_error_imag += val_errors['imag']

                self.evaluator.store_epoch_val_loss(epoch_val_loss)
                self.evaluator.store_epoch_val_real_error(epoch_val_error_real)
                self.evaluator.store_epoch_val_imag_error(epoch_val_error_imag)

                if epoch_val_error_real < best_avg_error_real:
                    best_avg_error_real = epoch_val_error_real
                    best_model_checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                    }
                    self.saver.save_model(best_model_checkpoint, 'best_model.pth')

                if epoch % 500 == 0:
                    print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.3E}, Val Loss: {epoch_val_loss:.3E}, Val Error Real: {epoch_val_error_real:.3E}")

            else:
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.3E}, Train Error Real: {epoch_train_error_real:.3E}, Train Error Imag: {epoch_train_error_imag:.3E}")

            self.training_strategy.after_epoch(epoch, self.model, self.p)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training concluded in: {training_time:.2f} seconds")

        epochs_plot = list(range(epochs))
        history = self.evaluator.get_history()
        fig = plot_training(epochs_plot, history)

        self.saver.save_outputs(
            model_state=best_model_checkpoint,
            model_info=self.p,
            split_indices=self.p['TRAIN_INDICES'],
            norm_params=self.p['NORMALIZATION_PARAMETERS'],
            history=history,
            figure=fig,
            time=training_time,
            figure_prefix="training_history",
            time_prefix="training"
        )
