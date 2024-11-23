import os
import yaml
import torch

from .model.deeponet import DeepONet
from .model.output_handling_strategies import (
    SingleTrunkSplitBranchStrategy,
    MultipleTrunksSplitBranchStrategy,
    SingleTrunkMultipleBranchesStrategy,
    MultipleTrunksMultipleBranchesStrategy
)
from .model.training_strategies import (
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)

def create_model(model_params, **kwargs):
    """Creates model from defined parameters.

    Args:
        model_params (dict): Model parameters (architecture, activation function, optimizer, etc.)

    Raises:
        ValueError: Raised when branch activation function is undefined.
        ValueError: Raised when trunk activation function is undefined.

    Returns:
        torch.nn.Module: Initialized model ready for training.
    """

    def get_activation_function(name):
        """
        Maps a string name to the corresponding PyTorch activation function.

        Args:
            name (str): Name of the activation function (e.g., 'relu', 'tanh').

        Returns:
            torch.nn.Module: Corresponding PyTorch activation function.

        Raises:
            ValueError: If the activation function name is not recognized.
        """
        activation_map = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid(),
            'leaky_relu': torch.nn.LeakyReLU(),
            'elu': torch.nn.ELU(),
            'gelu': torch.nn.GELU(),
            'softplus': torch.nn.Softplus(),
            'identity': torch.nn.Identity()
        }
        if name not in activation_map:
            raise ValueError(f"Unsupported activation function: '{name}'. Supported functions are: {list(activation_map.keys())}")
        return activation_map[name]
    
    if model_params.get('TWO_STEP_TRAINING', False) and model_params.get('PROPER_ORTHOGONAL_DECOMPOSITION', False):
        raise ValueError("Invalid configuration: POD and Two-Step Training cannot be enabled simultaneously.")
    
    def get_output_sizes(output_handling, basis_functions, n_outputs, trunk_output_size):
        """
        Computes the branch and trunk output sizes based on the output handling strategy.

        Args:
            output_handling (str): The output handling strategy (e.g., 'single_trunk_split_branch').
            basis_functions (int): Number of basis functions in the model.
            n_outputs (int): Number of outputs in the model.
            trunk_output_size (int): Default trunk output size.

        Returns:
            tuple: (branch_output_size, trunk_output_size)
        """
        if output_handling == 'single_trunk_split_branch':
            branch_output_size = basis_functions * n_outputs
            trunk_output_size = trunk_output_size
        elif output_handling == 'multiple_trunks_split_branch':
            branch_output_size = basis_functions
            trunk_output_size = trunk_output_size
        elif output_handling == 'single_trunk_multiple_branches':
            branch_output_size = basis_functions
            trunk_output_size = basis_functions * n_outputs
        elif output_handling == 'multiple_trunks_multiple_branches':
            branch_output_size = basis_functions
            trunk_output_size = basis_functions
        else:
            raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")
        return branch_output_size, trunk_output_size


    model_name = model_params.get('MODELNAME', 'DeepONet')
    if model_params.get('TWO_STEP_TRAINING', False):
        model_name += '_two_step'
    if model_params.get('PROPER_ORTHOGONAL_DECOMPOSITION', False):
        model_name += '_pod'
    
    else:
        model_params['BASIS_FUNCTIONS'] = model_params['TRUNK_OUTPUT_SIZE']

    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        model_params['TRUNK_INPUT_SIZE'] += 2 * model_params['TRUNK_INPUT_SIZE'] * model_params['TRUNK_EXPANSION_FEATURES_NUMBER']

    branch_output_size = model_params['BASIS_FUNCTIONS'] * model_params['N_OUTPUTS']
    
    if model_params.get('TRAINING_STRATEGY') == 'pod':
        branch_output_size = model_params['BASIS_FUNCTIONS']

    branch_output_size, trunk_output_size = get_output_sizes(
        output_handling=model_params.get('OUTPUT_HANDLING', 'single_trunk_split_branch').lower(),
        basis_functions=model_params['BASIS_FUNCTIONS'],
        n_outputs=model_params['N_OUTPUTS'],
        trunk_output_size=model_params['TRUNK_OUTPUT_SIZE']
    )


    model_params['BRANCH_LAYERS'] = (
        [model_params['BRANCH_INPUT_SIZE']] +
        model_params['BRANCH_HIDDEN_LAYERS'] +
        [branch_output_size]
    )
    model_params['TRUNK_LAYERS'] = (
        [model_params['TRUNK_INPUT_SIZE']] +
        model_params['TRUNK_HIDDEN_LAYERS'] +
        [trunk_output_size]
    )

    branch_config = {
        'architecture': model_params['BRANCH_ARCHITECTURE'],
        'layers': model_params['BRANCH_LAYERS'],
        'activation': get_activation_function(model_params.get('BRANCH_ACTIVATION', 'identity'))
    }

    trunk_config = {
        'architecture': model_params['TRUNK_ARCHITECTURE'],
        'layers': model_params['TRUNK_LAYERS'],
        'activation': get_activation_function(model_params.get('TRUNK_ACTIVATION', 'identity'))
    }

    n_outputs = model_params['N_OUTPUTS']
    output_handling = model_params.get('OUTPUT_HANDLING', 'single_trunk_split_branch').lower()

    if output_handling == 'single_trunk_split_branch':
        output_strategy = SingleTrunkSplitBranchStrategy()
    elif output_handling == 'multiple_trunks_split_branch':
        output_strategy = MultipleTrunksSplitBranchStrategy()
    elif output_handling == 'single_trunk_multiple_branches':
        output_strategy = SingleTrunkMultipleBranchesStrategy()
    elif output_handling == 'multiple_trunks_multiple_branches':
        output_strategy = MultipleTrunksMultipleBranchesStrategy()
    else:
        raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")
    training_strategy_name = model_params.get('TRAINING_STRATEGY', 'standard').lower()
    if training_strategy_name == 'pod':
        pod_basis = kwargs.get('pod_basis')
        mean_functions = kwargs.get('mean_functions')
        if pod_basis is None or mean_functions is None:
            raise ValueError("POD basis and mean functions must be provided for PODTrainingStrategy.")
        training_strategy = PODTrainingStrategy(pod_basis=pod_basis, mean_functions=mean_functions)
    elif training_strategy_name == 'two_step':
        A_dim = model_params.get('A_DIM')
        if A_dim is None:
            raise ValueError("A_DIM must be provided for TwoStepTrainingStrategy.")
        training_strategy = TwoStepTrainingStrategy(A_dim=A_dim, n_outputs=n_outputs)
    elif training_strategy_name == 'standard':
        training_strategy = StandardTrainingStrategy()
    else:
        raise ValueError(f"Unsupported TRAINING_STRATEGY: {training_strategy_name}")
    
    model = DeepONet(
        branch_config=branch_config,
        trunk_config=trunk_config,
        output_strategy=output_strategy,
        training_strategy=training_strategy,
        n_outputs=n_outputs
    ).to(model_params['DEVICE'], dtype=getattr(torch, model_params['PRECISION']))

    
    if model_params.get('INPUT_NORMALIZATION', False):
        model_name += '_in'
    if model_params.get('OUTPUT_NORMALIZATION', False):
        model_name += '_out'
    if model_params.get('INPUT_NORMALIZATION', False) or model_params['OUTPUT_NORMALIZATION']:
        model_name += '_norm'
    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        model_name += '_trunk_expansion'
    return model, model_params['MODELNAME']


def initialize_model(model_folder, model_name, device, precision):
    """
    Initializes and returns the model based on the saved configuration and state.

    Args:
        model_folder (str): Path to the folder containing the model and config files.
        model_name (str): Name of the model to load.
        device (str): Device to load the model on (e.g'cpu', 'cuda').
        precision (str): Precision for the model parameters (e.g., 'float32').

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
    
    n_outputs = config['N_OUTPUTS']
    output_handling = config.get('OUTPUT_HANDLING', 'single_trunk_split_branch').lower()

    if output_handling == 'single_trunk_split_branch':
        output_strategy = SingleTrunkSplitBranchStrategy()
    elif output_handling == 'multiple_trunks_split_branch':
        output_strategy = MultipleTrunksSplitBranchStrategy()
    elif output_handling == 'single_trunk_multiple_branches':
        output_strategy = SingleTrunkMultipleBranchesStrategy()
    elif output_handling == 'multiple_trunks_multiple_branches':
        output_strategy = MultipleTrunksMultipleBranchesStrategy()
    else:
        raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")

    training_strategy_name = config.get('TRAINING_STRATEGY', 'standard').lower()
    if training_strategy_name == 'pod':
        pod_basis = config.get('pod_basis')
        mean_functions = config.get('mean_functions')
        if pod_basis is None or mean_functions is None:
            raise ValueError("POD basis and mean functions must be provided for PODTrainingStrategy.")
        training_strategy = PODTrainingStrategy(pod_basis=pod_basis, mean_functions=mean_functions)
    elif training_strategy_name == 'two_step':
        A_dim = config.get('A_DIM')
        if A_dim is None:
            raise ValueError("A_DIM must be provided for TwoStepTrainingStrategy.")
        training_strategy = TwoStepTrainingStrategy(A_dim=A_dim, n_outputs=n_outputs)
    elif training_strategy_name == 'standard':
        training_strategy = StandardTrainingStrategy()
    else:
        raise ValueError(f"Unsupported TRAINING_STRATEGY: {training_strategy_name}")
    
    model = DeepONet(
        branch_config=branch_config,
        trunk_config=trunk_config,
        output_strategy=output_strategy,
        training_strategy=training_strategy,
        n_outputs=n_outputs
    ).to(device, dtype=getattr(torch, config['PRECISION']))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if config.get('TWO_STEP_TRAINING', False):
        Q_matrix = checkpoint['Q']
        R_matrix = checkpoint['R']
        T_matrix = checkpoint['T']
        model.set_Q(Q_matrix.to(device, precision))
        model.set_R(R_matrix.to(device, precision))
        model.set_T(T_matrix.to(device, precision))

    if config.get('PROPER_ORTHOGONAL_DECOMPOSITION', False):
        POD_matrix = checkpoint['POD_basis']
        mean_fns = checkpoint['mean_functions']
        model.get_basis(POD_matrix.to(device, precision))
        model.get_mean_functions(mean_fns.to(device, precision))
    
    model.eval()
    
    return model, config