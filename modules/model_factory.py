import os
import yaml
import torch
from modules.deeponet import DeepONet
from modules.deeponet_two_step import DeepONetTwoStep

def create_model(model_params):
    """Creates model from defined parameters.

    Args:
        model_params (dict): Model parameters (architecture, activation function, optimizer, etc.)

    Raises:
        ValueError: Raised when branch activation function is undefined.
        ValueError: Raised when trunk activation function is undefined.

    Returns:
        torch.nn.Module: Initialized model ready for training.
    """
    if not model_params['TWO_STEP_TRAINING']:
        model_params['BASIS_FUNCTIONS'] = model_params['TRUNK_OUTPUT_SIZE']
    else:
        model_params['MODELNAME'] += '_two_step'
        model_params['TRUNK_OUTPUT_SIZE'] = model_params['BASIS_FUNCTIONS']

    if model_params['TRUNK_FEATURE_EXPANSION']:
        model_params['TRUNK_INPUT_SIZE'] += 2 * model_params['TRUNK_INPUT_SIZE'] * model_params['TRUNK_EXPANSION_FEATURES_NUMBER']

    model_params['BRANCH_LAYERS'] = [model_params['BRANCH_INPUT_SIZE']] + model_params['BRANCH_HIDDEN_LAYERS'] + [model_params['BASIS_FUNCTIONS'] * model_params['N_OUTPUTS']]
    model_params['TRUNK_LAYERS'] = [model_params['TRUNK_INPUT_SIZE']] + model_params['TRUNK_HIDDEN_LAYERS'] + [model_params['TRUNK_OUTPUT_SIZE']]

    branch_config = {
        'architecture': model_params['BRANCH_ARCHITECTURE'],
        'layers': model_params['BRANCH_LAYERS'],
    }

    trunk_config = {
        'architecture': model_params['TRUNK_ARCHITECTURE'],
        'layers': model_params['TRUNK_LAYERS'],
    }

    if model_params['BRANCH_ARCHITECTURE'].lower() == 'mlp':
        model_params['MODELNAME'] += '_mlp_'
        try:
            if model_params['BRANCH_MLP_ACTIVATION'].lower() == 'relu':
                branch_activation = torch.nn.ReLU()
            elif model_params['BRANCH_MLP_ACTIVATION'].lower() == 'leaky_relu':
                branch_activation = torch.nn.LeakyReLU(negative_slope=0.01)
            elif model_params['BRANCH_MLP_ACTIVATION'].lower() == 'tanh':
                branch_activation = torch.tanh
            else:
                raise ValueError
        except ValueError:
            print('Invalid activation function for branch net.')
        branch_config['activation'] = branch_activation
    elif model_params['BRANCH_ARCHITECTURE'].lower() == 'resnet':
        model_params['MODELNAME'] += '_resnet_'
        try:
            if model_params['BRANCH_MLP_ACTIVATION'].lower() == 'relu':
                branch_activation = torch.nn.ReLU()
            elif model_params['BRANCH_MLP_ACTIVATION'].lower() == 'leaky_relu':
                branch_activation = torch.nn.LeakyReLU(negative_slope=0.01)
            elif model_params['BRANCH_MLP_ACTIVATION'].lower() == 'tanh':
                branch_activation = torch.tanh
            else:
                raise ValueError
        except ValueError:
            print('Invalid activation function for branch net.')
        branch_config['activation'] = branch_activation
    else:
        model_params['MODELNAME'] += '_kan_'
        branch_config['degree'] = model_params['BRANCH_KAN_DEGREE']

    if model_params['TRUNK_ARCHITECTURE'].lower() == 'mlp':
        model_params['MODELNAME'] += 'mlp'
        try:
            if model_params['TRUNK_MLP_ACTIVATION'].lower() == 'relu':
                trunk_activation = torch.nn.ReLU()
            elif model_params['TRUNK_MLP_ACTIVATION'].lower() == 'leaky_relu':
                trunk_activation = torch.nn.LeakyReLU(negative_slope=0.01)
            elif model_params['TRUNK_MLP_ACTIVATION'].lower() == 'tanh':
                trunk_activation = torch.tanh
            else:
                raise ValueError
        except ValueError:
            print('Invalid activation function for trunk net.')
        trunk_config['activation'] = trunk_activation
    elif model_params['TRUNK_ARCHITECTURE'].lower() == 'resnet':
        model_params['MODELNAME'] += 'resnet'
        try:
            if model_params['TRUNK_MLP_ACTIVATION'].lower() == 'relu':
                trunk_activation = torch.nn.ReLU()
            elif model_params['TRUNK_MLP_ACTIVATION'].lower() == 'leaky_relu':
                trunk_activation = torch.nn.LeakyReLU(negative_slope=0.01)
            elif model_params['TRUNK_MLP_ACTIVATION'].lower() == 'tanh':
                trunk_activation = torch.tanh
            else:
                raise ValueError
        except ValueError:
            print('Invalid activation function for trunk net.')
        trunk_config['activation'] = trunk_activation
    else:
        model_params['MODELNAME'] += 'kan'
        trunk_config['degree'] = model_params['TRUNK_KAN_DEGREE']

    if model_params['TWO_STEP_TRAINING']:
        model = DeepONetTwoStep(branch_config=branch_config,
                        trunk_config=trunk_config,
                        A_dim=(model_params['N_OUTPUTS'], model_params["BASIS_FUNCTIONS"], len(model_params['TRAIN_INDICES']))
                        ).to(model_params['DEVICE'], eval(model_params['PRECISION']))
    else:
        model = DeepONet(branch_config=branch_config,
                        trunk_config=trunk_config,
                        ).to(model_params['DEVICE'], eval(model_params['PRECISION']))
    
    if model_params['INPUT_NORMALIZATION']:
        model_params['MODELNAME'] += '_input'
    if model_params['OUTPUT_NORMALIZATION']:
        model_params['MODELNAME'] += '_output'
    if model_params['INPUT_NORMALIZATION'] or model_params['OUTPUT_NORMALIZATION']:
        model_params['MODELNAME'] += '_normalization'
    if model_params['TRUNK_FEATURE_EXPANSION']:
        model_params['MODELNAME'] += '_feature_expansion'
    return model, model_params['MODELNAME']

def initialize_model(model_folder, model_name, device, precision):
    """
    Initializes and returns the model based on the saved configuration and state.

    Args:
        model_folder (str): Path to the folder containing the model and config files.
        model_name (str): Name of the model to load.
        device (str): Device to load the model on (e.g'cpu', 'cuda').
        precision (torch.dtype): Precision for the model parameters.

    Returns:
        torch.nn.Module: The initialized model ready for inference.
        dict: The loaded configuration parameters.
    """
    
    model_path = os.path.join(model_folder, f"model_state_{model_name}.pth")
    config_path = os.path.join(model_folder, f"model_info_{model_name}.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    branch_activation_str = config.get('BRANCH_MLP_ACTIVATION', 'relu').lower()
    if branch_activation_str == 'relu':
        branch_activation = torch.nn.ReLU()
    elif branch_activation_str == 'leaky_relu':
        branch_activation = torch.nn.LeakyReLU(negative_slope=0.01)
    elif branch_activation_str == 'tanh':
        branch_activation = torch.nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {branch_activation_str}")
    
    trunk_activation_str = config.get('TRUNK_MLP_ACTIVATION', 'relu').lower()
    if trunk_activation_str == 'relu':
        trunk_activation = torch.nn.ReLU()
    elif trunk_activation_str == 'leaky_relu':
        trunk_activation = torch.nn.LeakyReLU(negative_slope=0.01)
    elif trunk_activation_str == 'tanh':
        trunk_activation = torch.nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {trunk_activation_str}")
    
    branch_config = {
        'architecture': config['BRANCH_ARCHITECTURE'],
        'layers': config["BRANCH_LAYERS"],
        'activation': branch_activation
    }
    
    trunk_config = {
        'architecture': config['TRUNK_ARCHITECTURE'],
        'layers': config["TRUNK_LAYERS"],
        'activation': trunk_activation
    }
    
    if config['TWO_STEP_TRAINING']:
        model = DeepONetTwoStep(
            branch_config=branch_config,
            trunk_config=trunk_config,
            A_dim=(config['N_OUTPUTS'], config['BASIS_FUNCTIONS'], len(config['TRAIN_INDICES']))
        ).to(device, precision)
    else:
        model = DeepONet(
            branch_config=branch_config,
            trunk_config=trunk_config
        ).to(device, precision)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.eval()
    
    return model, config