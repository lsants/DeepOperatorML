import os
import json
import torch
import numpy as np

class Saver:
    def __init__(self, model_name, model_folder=None, data_output_folder=None, figures_folder=None):
        self.name = model_name
        self.model_folder = model_folder
        self.data_output_folder = data_output_folder
        self.figures_folder = figures_folder

    def __call__(self, model_state_dict=None, split_indices=None, norm_params=None, history=None, figure=None, errors=None, time=None, figure_suffix=None, time_prefix=None):
        if model_state_dict is not None:
            self.save_model(model_state_dict)
        if split_indices is not None:
            self.save_indices(split_indices)
        if norm_params is not None:
            self.save_norm_params(norm_params)
        if history is not None:
            self.save_history(history)
        if figure is not None:
            self.save_plots(figure, figure_suffix)
        if errors is not None:
            self.save_errors(errors)
        if time is not None:
            self.save_time(time, time_prefix)

    def save_checkpoint(self, model_state_dict, optimizer_state_dict, epoch):
        filename = f'model_checkpoint_{self.name}_epoch{epoch}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'epoch': epoch
        }, model_path)
        print(f"Checkpoint saved to {model_path}\n")

    def save_model(self, model_state):
        filename = f'model_state_{self.name}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}\n")

    def save_indices(self, indices_dict):
        filename = f'indices_{self.name}.json'
        indices_path = self.make_output_dir(self.data_output_folder, filename)
        with open(indices_path, 'w') as f:
            json.dump(indices_dict, f)
        print(f"Indices saved to {indices_path}\n")

    def save_norm_params(self, norm_params_dict):
        filename = f'norm_params_{self.name}.json'
        norm_params_path = self.make_output_dir(self.data_output_folder, filename)
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params_dict, f)
        print(f"Normalization parameters saved to {norm_params_path}\n")

    def save_history(self, history_dict):
        filename = f'history_{self.name}.json'
        history_path = self.make_output_dir(self.data_output_folder, filename)
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
        print(f"Training history saved to {history_path}\n")

    def save_plots(self, figure, filename_suffix=None):
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        filename = f'plot_{self.name}{suffix}.png'
        fig_path = self.make_output_dir(self.figures_folder, filename)
        figure.savefig(fig_path)
        print(f"Figure saved to {fig_path}\n")

    def save_errors(self, errors_dict):
        filename = f"errors_{self.name}.json"
        errors_path = os.path.join(self.data_output_folder, filename)
        errors_serializable = {k: float(v) if isinstance(v, np.ndarray) else v for k, v in errors_dict.items()}
    
        with open(errors_path, "w") as f:
            json.dump(errors_serializable, f)
        print(f"Saved errors to {errors_path}\n")

    def save_time(self, time_dict, filename_prefix=None):
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{prefix}time_{self.name}.json"
        time_path = os.path.join(self.data_output_folder, filename)
        time_serializable = {k: float(v) if isinstance(v, np.ndarray) else v for k, v in time_dict.items()}
    
        with open(time_path, "w") as f:
            json.dump(time_serializable, f)
        print(f"Saved time to {time_path}\n")

    def make_output_dir(self, folder, filename):
        if folder is None:
            raise ValueError(f"The specified folder for saving '{filename}' is None.")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        return os.path.join(folder, filename)
