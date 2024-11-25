import os
import yaml
import torch
import numpy as np


class Saver:
    def __init__(self, model_name, model_folder=None, data_output_folder=None, figures_folder=None):
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

    def __call__(self, phase=None, **kwargs):
        """
        Saves various components based on provided keyword arguments.

        Args:
            phase (str, optional): The current training phase (e.g., 'trunk', 'branch'). Defaults to None.
            **kwargs: Arbitrary keyword arguments corresponding to components to save.
        """
        for key, value in kwargs.items():
            if key in self.save_methods:
                if key == 'history':
                    self.save_methods[key](value, phase=phase, filename_prefix=kwargs.get('history_prefix'))
                elif key == 'figure':
                    self.save_methods[key](value, phase=phase, filename_prefix=kwargs.get('figure_prefix'))
                elif key == 'time':
                    self.save_methods[key](value, phase=phase, filename_prefix=kwargs.get('time_prefix'))
                elif key == 'train_state':
                    self.save_methods[key](value['model_state_dict'],
                                           value['optimizer_state_dict'],
                                           value['epochs'],
                                           phase=phase)
                else:
                    self.save_methods[key](value, phase=phase)

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

    def save_checkpoint(self, model_state_dict, optimizer_state_dict, epoch, phase=None):
        filename = f'{phase or "default"}_checkpoint_{self.name}_epoch_{epoch}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'epoch': epoch
        }, model_path)
        print(f"Checkpoint saved to {model_path}\n")

    def save_model(self, model_state, phase=None):
        filename = f'{phase or "default"}_model_state_{self.name}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)

        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}\n")

    def save_model_info(self, model_info_dict, phase=None):
        filename = f'{phase or "default"}_model_info_{self.name}.yaml'
        model_info_path = self.make_output_dir(self.model_folder, filename)
        serializable_model_info = self.make_serializable(model_info_dict)
        with open(model_info_path, 'w') as f:
            yaml.dump(serializable_model_info, f)
        print(f"Model information saved to {model_info_path}\n")

    def save_indices(self, indices_dict, phase=None):
        filename = f'{phase or "default"}_indices_{self.name}.yaml'
        indices_path = self.make_output_dir(self.data_output_folder, filename)
        with open(indices_path, 'w') as f:
            yaml.dump(indices_dict, f)
        print(f"Indices saved to {indices_path}\n")

    def save_norm_params(self, norm_params_dict, phase=None):
        filename = f'{phase or "default"}_norm_params_{self.name}.yaml'
        norm_params_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_norm_params = self.make_serializable(norm_params_dict)
        with open(norm_params_path, 'w') as f:
            yaml.dump(serializable_norm_params, f, indent=4)
        print(f"Normalization parameters saved to {norm_params_path}\n")

    def save_history(self, history_dict, phase=None, filename_prefix=None):
        filename = f'{phase or "default"}_{filename_prefix or "history"}_{self.name}.yaml'
        history_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_history = self.make_serializable(history_dict)
        with open(history_path, 'w') as f:
            yaml.dump(serializable_history, f, indent=4)
        print(f"Training history saved to {history_path}\n")

    def save_plots(self, figure, phase=None, filename_prefix=None):
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f'{phase or "default"}_{prefix}plot_{self.name}.png'
        fig_path = self.make_output_dir(self.figures_folder, filename)
        figure.savefig(fig_path)
        print(f"Figure saved to {fig_path}\n")

    def save_errors(self, errors_dict, phase=None):
        filename = f"{phase or 'default'}_errors_{self.name}.yaml"
        errors_path = self.make_output_dir(self.data_output_folder, filename)
        errors_serializable = self.make_serializable(errors_dict)
        with open(errors_path, "w") as f:
            yaml.dump(errors_serializable, f, indent=4)
        print(f"Errors saved to {errors_path}\n")

    def save_time(self, time_dict, phase=None, filename_prefix=None):
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{phase or 'default'}_{prefix}time_{self.name}.yaml"
        time_path = self.make_output_dir(self.data_output_folder, filename)
        time_serializable = self.make_serializable(time_dict)
        with open(time_path, "w") as f:
            yaml.dump(time_serializable, f, indent=4)
        print(f"Time information saved to {time_path}\n")

    def make_output_dir(self, folder, filename):
        """Ensures that the output directory exists and returns the full file path."""
        if folder is None:
            raise ValueError(f"The specified folder for saving '{filename}' is undefined.")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)
