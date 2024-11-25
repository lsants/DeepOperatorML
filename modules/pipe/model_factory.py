import os
import yaml
import torch

from ..deeponet.deeponet import DeepONet
from ..deeponet.training_strategies import (
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)
from ..deeponet.output_strategies import (
    SingleTrunkSplitBranchStrategy,
    MultipleTrunksSplitBranchStrategy,
    SingleTrunkMultipleBranchesStrategy,
    MultipleTrunksMultipleBranchesStrategy
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
    
    model_name = model_params.get('MODELNAME', 'DeepONet')

    trunk_input_size = model_params['TRUNK_INPUT_SIZE']
    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        trunk_input_size += (
            2 * model_params['TRUNK_INPUT_SIZE'] * model_params['TRUNK_EXPANSION_FEATURES_NUMBER']
        )

    branch_architecture = model_params['BRANCH_ARCHITECTURE']
    trunk_architecture = model_params['TRUNK_ARCHITECTURE']
    
    basis_functions = model_params.get('BASIS_FUNCTIONS')

    branch_config = {
        'architecture': branch_architecture,
        'layers': (
            [model_params['BRANCH_INPUT_SIZE']]
            + model_params['BRANCH_HIDDEN_LAYERS']
            + [model_params['BRANCH_HIDDEN_LAYERS'][-1]]
        ),
    }

    trunk_config = {
        'architecture': trunk_architecture,
        'layers': (
            [trunk_input_size]
            + model_params['TRUNK_HIDDEN_LAYERS']
            + [model_params['TRUNK_HIDDEN_LAYERS'][-1]]
        ),
    }

    if branch_architecture.lower() == 'mlp':
        branch_config['activation'] = get_activation_function(model_params.get('BRANCH_ACTIVATION'))

    if trunk_architecture.lower() == 'mlp':
        trunk_config['activation'] = get_activation_function(model_params.get('TRUNK_ACTIVATION'))

    if branch_architecture.lower() == 'resnet':
        branch_config['activation'] = get_activation_function(model_params.get('BRANCH_ACTIVATION'))

    if trunk_architecture.lower() == 'resnet':
        trunk_config['activation'] = get_activation_function(model_params.get('TRUNK_ACTIVATION'))

    if branch_architecture.lower() == 'kan':
        branch_config['degree'] = model_params.get('BRANCH_DEGREE')

    if trunk_architecture.lower() == 'kan':
        trunk_config['degree'] = model_params.get('TRUNK_DEGREE')

    output_handling = model_params.get('OUTPUT_HANDLING', 'single_trunk_split_branch').lower()
    output_strategy_mapping = {
        'single_trunk_split_branch': SingleTrunkSplitBranchStrategy,
        'multiple_trunks_split_branch': MultipleTrunksSplitBranchStrategy,
        'single_trunk_multiple_branches': SingleTrunkMultipleBranchesStrategy,
        'multiple_trunks_multiple_branches': MultipleTrunksMultipleBranchesStrategy,
    }

    if output_handling not in output_strategy_mapping:
        raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")

    output_strategy = output_strategy_mapping[output_handling]()

    training_strategy_name = model_params.get('TRAINING_STRATEGY').lower()

    if training_strategy_name == 'pod':
        pod_basis = kwargs.get('pod_basis')
        mean_functions = kwargs.get('mean_functions')
        if pod_basis is None or mean_functions is None:
            raise ValueError("POD basis and mean functions must be provided for PODTrainingStrategy.")
        training_strategy = PODTrainingStrategy(pod_basis=pod_basis, mean_functions=mean_functions)

    elif training_strategy_name == 'two_step':
        batch_size = kwargs.get('train_dataset_length')
        if batch_size is None:
            training_strategy = TwoStepTrainingStrategy()
        else:
            training_strategy = TwoStepTrainingStrategy(train_dataset_length=batch_size)

    elif training_strategy_name == 'standard':
        training_strategy = StandardTrainingStrategy()

    else:
        raise ValueError(f"Unsupported TRAINING_STRATEGY: {training_strategy_name}")

    model = DeepONet(
        branch_config=branch_config,
        trunk_config=trunk_config,
        output_strategy=output_strategy,
        training_strategy=training_strategy,
        n_outputs=model_params['N_OUTPUTS'],
        basis_functions=basis_functions,
    ).to(model_params['DEVICE'], dtype=getattr(torch, model_params['PRECISION']))
    
    if model_params.get('TRAINING_STRATEGY', False):
        model_name += '_' + model_params.get('TRAINING_STRATEGY')
    if model_params.get('INPUT_NORMALIZATION', False):
        model_name += '_in'
    if model_params.get('OUTPUT_NORMALIZATION', False):
        model_name += '_out'
    if model_params.get('INPUT_NORMALIZATION', False) or model_params['OUTPUT_NORMALIZATION']:
        model_name += '_norm'
    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        model_name += '_trunk_expansion'

    return model, model_name

def initialize_model(model_folder, model_name, device, precision):
    """
    Initializes and returns the model based on the saved configuration and state.

    Args:
        model_folder (str): Path to the folder containing the model and config files.
        model_name (str): Name of the model to load.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').
        precision (str): Precision for the model parameters (e.g., 'float32').

    Returns:
        torch.nn.Module: The initialized model ready for inference.
        dict: The loaded configuration parameters.
    """
    model_path = os.path.join(model_folder, f"model_state_{model_name}.pth")
    config_path = os.path.join(model_folder, f"model_info_{model_name}.yaml")

    
    with open(config_path, 'r') as file:
        model_params = yaml.safe_load(file)

    model_params['DEVICE'] = device
    model_params['PRECISION'] = precision

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    pod_basis = checkpoint.get('POD_basis', None)
    mean_functions = checkpoint.get('mean_functions', None)

    model, _ = create_model(
        model_params,
        pod_basis=pod_basis,
        mean_functions=mean_functions,
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    training_strategy = model_params.get('TRAINING_STRATEGY', '').lower()
    if training_strategy == 'two_step':
        print(checkpoint.keys())
        model.training_strategy.set_matrices(Q_list=checkpoint.get('Q'),
                                                     R_list=checkpoint.get('R'),
                                                     T_list=checkpoint.get('T'))
    elif training_strategy == 'pod':
        if pod_basis is not None:
            model.get_basis(pod_basis.to(device, dtype=getattr(torch, precision)))
        if mean_functions is not None:
            model.get_mean_functions(mean_functions.to(device, dtype=getattr(torch, precision)))
    
    model.training_strategy.inference_mode()
    model.eval()
    return model, model_params
