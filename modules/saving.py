import os
import json
import torch
import matplotlib.pyplot as plt

# To do: implement saving on checkpoints

class Saver:
    def __init__(self, model_name, model_folder=None, data_output_folder=None, figures_folder=None):
        self.name = model_name
        self.model_folder = model_folder
        self.data_output_folder = data_output_folder
        self.figures_folder = figures_folder

    def __call__(self, model_state_dict=None, split_indices=None, norm_params=None, history=None, figure=None):
        if model_state_dict:
            self.save_model(model_state_dict)
        if split_indices:
            self.save_indices(split_indices)
        if norm_params:
            self.save_norm_params(norm_params)
        if history:
            self.save_history(history)
        if figure:
            self.save_plots(figure)
    
    def save_checkpoint(self, model_state_dict, optimizer_state_dict, epoch):
            torch.save({
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer_state_dict,
                        'epoch': epoch
            }, self.model_path)

    def save_model(self, model_state):
        output = 'model_state_'
        extension = '.pth'
        filename = output + self.name + extension
        if not self.model_folder:
            self.make_output_dir(self.model_folder, filename)

        model_path = os.path.join(self.model_folder, filename)
        torch.save(model_state, model_path)
        print(f"Saved model to {model_path}")

    def save_indices(self, indices_dict):
        output = 'indices_'
        extension = '.json'
        filename = output + self.name + extension
        if not self.data_output_folder:
            self.make_output_dir(self.data_output_folder, filename)

        indices_path = os.path.join(self.data_output_folder, filename)
        with open(indices_path, 'w') as f:
            json.dump(indices_dict, f)
        print(f"Saved indices to {indices_path}")

    def save_norm_params(self, norm_params_dict):
        output = 'norm_params_'
        extension = '.json'
        filename = output + self.name + extension
        if not self.data_output_folder:
            self.make_output_dir(self.data_output_folder, filename)

        norm_params_path = os.path.join(self.data_output_folder, filename)
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params_dict, f)
        print(f"Saved indices to {norm_params_path}")

    def save_history(self, history_dict):
        output = 'history_'
        extension = '.json'
        filename = output + self.name + extension
        if not self.data_output_folder:
            self.make_output_dir(self.data_output_folder, filename)

        history_path = os.path.join(self.data_output_folder, filename)
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
        print(f"Saved history to {history_path}")

    def save_plots(self, figure):
        output = 'history_plot_'
        extension = '.png'
        filename = output + self.name + extension
        if not self.figures_folder:
            self.make_output_dir(self.figures_folder, filename)

        fig_path = os.path.join(self.figures_folder, filename)
        figure.savefig(fig_path)
        print(f"Saved figure to {fig_path}")

    def make_output_dir(self, folder, filename):
        fname = os.path.join(folder, filename)
        output_folder = fname

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            output_folder = output_folder + "_copy"
            os.makedirs(output_folder, exist_ok=True)

        return output_folder