import os
import yaml
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
class Saver:
    def __init__(self, model_name: str, model_folder: str | None=None, data_output_folder: str | None=None, figures_folder: str | None=None, full_logging: bool=True):
        """
        Initializes the Saver with designated folders for models, data, and figures.

        Args:
            model_name (str): The name of the model, used for file naming.
            model_folder (str, optional): Directory to save model state dictionaries. Defaults to None.
            data_output_folder (str, optional): Directory to save data-related files (e.g., indices, normalization params). Defaults to None.
            figures_folder (str, optional): Directory to save figures/plots. Defaults to None.
        """
        self.name = model_name
        self.model_folder = model_folder
        self.data_output_folder = data_output_folder
        self.figures_folder = figures_folder
        self.save_methods = {
            'train_state': self.save_checkpoint,
            'model_state': self.save_model,
            'model_info': self.save_model_info,
            'split_indices': self.save_indices,
            'norm_params': self.save_norm_params,
            'history': self.save_history,
            'figure': self.save_plots,
            'errors': self.save_errors,
            'time': self.save_time,
        }
        self.full_logging = full_logging

    def __call__(self, **kwargs):
        """
        Saves various components based on provided keyword arguments.

        Args:
            name (str, optional): The current name or phase (e.g., 'POD', trunk', 'branch'). Defaults to None.
            **kwargs: Arbitrary keyword arguments corresponding to components to save.
        """
        for key, value in kwargs.items():
            if key in self.save_methods:
                if key == 'history':
                    self.save_methods[key](value, filename_prefix=kwargs.get('history_prefix'))
                elif key == 'figure':
                    self.save_methods[key](value, filename_prefix=kwargs.get('figure_prefix'))
                elif key == 'time':
                    self.save_methods[key](value, filename_prefix=kwargs.get('time_prefix'))
                elif key == 'train_state':
                    self.save_methods[key](value['model_state_dict'],
                                           value['optimizer_state_dict'],
                                           value['epochs'],
                                        )
                else:
                    self.save_methods[key](value)

    def set_logging(self, logging: bool) -> None:
        self.full_logging = logging

    def make_serializable(self, obj):
        """Recursively converts non-serializable objects to serializable ones."""
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save_checkpoint(self, model_state_dict, optimizer_state_dict, epoch) -> None:
        filename = f'checkpoint_{self.name}_epoch_{epoch}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'epoch': epoch
        }, model_path)
        logger.info(f"\nCheckpoint saved to:\n{model_path}\n")

    def save_model(self, model_state,) -> None:
        filename = f'model_state_{self.name}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save(model_state, model_path)
        logger.info(f"\nModel saved to:\n{model_path}\n")

    def save_model_info(self, model_info: dict,) -> None:
        filename = f'model_info_{self.name}.yaml'
        model_info_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_model_info = self.make_serializable(model_info)
        with open(model_info_path, 'w') as f:
            yaml.dump(serializable_model_info, f)
        logger.info(f"\nModel information saved to:\n{model_info_path}\n")

    def save_indices(self, indices_dict,) -> None:
        filename = f'indices_{self.name}.yaml'
        indices_path = self.make_output_dir(self.data_output_folder, filename)
        with open(indices_path, 'w') as f:
            yaml.dump(indices_dict, f)
        if self.full_logging:
            logger.info(f"\nIndices saved to:\n{indices_path}\n")

    def save_norm_params(self, norm_params: dict,) -> None:
        filename = f'norm_params_{self.name}.yaml'
        norm_params_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_norm_params = self.make_serializable(norm_params)
        with open(norm_params_path, 'w') as f:
            yaml.dump(serializable_norm_params, f, indent=4)
        if self.full_logging:
            logger.info(f"\nNormalization parameters saved to:\n{norm_params_path}\n")

    def save_history(self, history: dict, filename_prefix=None) -> None:
        filename = f'{filename_prefix or "history"}_{self.name}.txt'
        history_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_history = self.make_serializable(history)
        with open(history_path, 'w') as f:
            yaml.dump(serializable_history, f, indent=4)
            logger.info(f"\nTraining history saved to:\n{history_path}\n")

    def save_plots(self, figure, filename_prefix=None) -> None:
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f'{prefix}plot_{self.name}.png'
        fig_path = self.make_output_dir(self.figures_folder, filename)
        figure.savefig(fig_path)
        if self.full_logging:
            logger.info(f"\nFigure saved to:\n{fig_path}\n")

    def save_errors(self, errors: dict,) -> None:
        filename = f"errors_{self.name}.txt"
        errors_path = self.make_output_dir(self.data_output_folder, filename)
        errors_serializable = self.make_serializable(errors)
        with open(errors_path, "w") as f:
            yaml.dump(errors_serializable, f, indent=4)
        if self.full_logging:
            logger.info(f"\nErrors saved to:\n{errors_path}\n")

    def save_time(self, times: dict, filename_prefix=None) -> None:
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{prefix}time_{self.name}.txt"
        time_path = self.make_output_dir(self.data_output_folder, filename)
        time_serializable = self.make_serializable(times)
        with open(time_path, "w") as f:
            yaml.dump(time_serializable, f, indent=4)
        if self.full_logging:
            logger.info(f"\nTime information saved to:\n{time_path}\n")

    def make_output_dir(self, folder: str, filename: str) -> str:
        """Ensures that the output directory exists and returns the full file path."""
        if folder is None:
            raise ValueError(f"The specified folder for saving '{filename}' is undefined.")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)