import os
import yaml
import torch
import numpy as np

class Saver:
    def __init__(self, model_name, model_folder=None, data_output_folder=None, figures_folder=None):
        """ Initializes the Saver with designated folders for models, data, and figures.

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

    def __call__(self, **kwargs):
        """
        Saves various components based on provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments corresponding to components to save.
                      Supported keys: 'model_state_dict', 'model_info', 'split_indices',
                      'norm_params', 'history', 'figure', 'errors', 'time'.
                      Each key maps to its respective save method.
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
                       value['epochs'])
                else:
                    self.save_methods[key](value)

    def make_serializable(self, obj):
        """
        Recursively converts non-serializable objects to serializable ones.

        Args:
            obj: The object to convert.

        Returns:
            A yaml-serializable version of the object.
        """
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
        """Saves the model's state dictionary.

        Args:
            model_state (dict): The state dictionary of the model.
        """
        filename = f'model_state_{self.name}.pth'
        model_path = self.make_output_dir(self.model_folder, filename)
        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}\n")

    def save_model_info(self, model_info_dict):
        """
        Saves the model's configuration and other relevant information.

        Args:
            model_info_dict (dict): The model information/configuration dictionary.
        """
        try:
            filename = f'model_info_{self.name}.yaml'
            model_info_path = self.make_output_dir(self.model_folder, filename)
            serializable_model_info = self.make_serializable(model_info_dict)
            
            with open(model_info_path, 'w') as f:
                yaml.dump(serializable_model_info, f)
            print(f"Model information saved to {model_info_path}\n")
        except Exception as e:
            print(f"Error saving model information: {e}")

    def save_indices(self, indices_dict):
        """Saves the dataset indices used for training and testing.

        Args:
            indices_dict (dict): Dictionary containing 'train' and 'test' indices.
        """
        filename = f'indices_{self.name}.yaml'
        indices_path = self.make_output_dir(self.data_output_folder, filename)
        with open(indices_path, 'w') as f:
            yaml.dump(indices_dict, f)
        print(f"Indices saved to {indices_path}\n")

    def save_norm_params(self, norm_params_dict):
        """Saves the normalization parameters for input and output data.

        Args:
            norm_params_dict (dict): Dictionary containing normalization parameters.
        """
        filename = f'norm_params_{self.name}.yaml'
        norm_params_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_norm_params = self.make_serializable(norm_params_dict)
        with open(norm_params_path, 'w') as f:
            yaml.dump(serializable_norm_params, f, indent=4)
        print(f"Normalization parameters saved to {norm_params_path}\n")

    def save_history(self, history_dict, filename_prefix=None):
        """Saves the training history

        Args:
            history_dict (dict): ictionary containing training history metrics.
            filename_prefix (str, optional): Prefix to add to the history filename (e.g., 'trunk', 'branch').
                                             Defaults to None.
        """
        filename = f'{filename_prefix}_history_{self.name}.yaml'
        history_path = self.make_output_dir(self.data_output_folder, filename)
        serializable_history = self.make_serializable(history_dict)
        
        with open(history_path, 'w') as f:
            yaml.dump(serializable_history, f, indent=4)
        print(f"Training history saved to {history_path}\n")

    def save_plots(self, figure, filename_prefix=None):
        """Saves a matplotlib figure.

        Args:
            figure (matplotlib.figure.Figure): The figure to save.
            filename_prefix (str, optional):  Prefix to add to the figure filename. Defaults to None.
        """
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f'{prefix}plot_{self.name}.png'
        fig_path = self.make_output_dir(self.figures_folder, filename)
        figure.savefig(fig_path)
        print(f"Figure saved to {fig_path}\n")

    def save_errors(self, errors_dict):
        """Saves the evaluation errors.

        Args:
            errors_dict (dict): Dictionary containing error metrics.
        """
        filename = f"errors_{self.name}.yaml"
        errors_path = self.make_output_dir(self.data_output_folder, filename)
        errors_serializable = self.make_serializable(errors_dict)
        
        with open(errors_path, "w") as f:
            yaml.dump(errors_serializable, f, indent=4)
        print(f"Errors saved to {errors_path}\n")

    def save_time(self, time_dict, filename_prefix=None):
        """Saves the inference/training time.

        Args:
            time_dict (dict): Dictionary containing timing information.
            filename_prefix (str, optional): Prefix to add to the time filename (e.g., 'inference'). Defaults to None.
        """
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{prefix}time_{self.name}.yaml"
        time_path = self.make_output_dir(self.data_output_folder, filename)
        time_serializable = self.make_serializable(time_dict)
        
        with open(time_path, "w") as f:
            yaml.dump(time_serializable, f, indent=4)
        print(f"Time information saved to {time_path}\n")

    def save_optimizer_state(self, optimizer_state_dict):
        """
        Saves the optimizer's state dictionary. (Optional for future use)

        Args:
            optimizer_state_dict (dict): The state dictionary of the optimizer.
        """
        filename = f'optimizer_state_{self.name}.pth'
        optimizer_path = self.make_output_dir(self.model_folder, filename)
        torch.save(optimizer_state_dict, optimizer_path)
        print(f"Optimizer state saved to {optimizer_path}\n")

    def make_output_dir(self, folder, filename):
        """Ensures that the output directory exists and returns the full file path.

        Args:
            folder (str): The directory to save the file.
            filename (str): The name of the file.

        Returns:
            str: Full path to the file.
        """
        if folder is None:
            raise ValueError(f"The specified folder for saving '{filename}' is undefined.")
        
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)
