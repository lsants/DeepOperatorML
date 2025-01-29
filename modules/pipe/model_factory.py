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
    SplitTrunkSingleBranchStrategy,
    MultipleTrunksSingleBranchStrategy,
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
    var_share = model_params.get('VAR_SHARE')
    data = kwargs.get('train_data')

    trunk_input_size = model_params['TRUNK_INPUT_SIZE']
    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        trunk_input_size += (
            2 * model_params['TRUNK_INPUT_SIZE'] * model_params['TRUNK_EXPANSION_FEATURES_NUMBER']
        )


    branch_architecture = model_params['BRANCH_ARCHITECTURE']
    trunk_architecture = model_params['TRUNK_ARCHITECTURE']

    branch_config = {
        'architecture': branch_architecture,
        'layers': (
            [model_params['BRANCH_INPUT_SIZE']]
            + model_params['BRANCH_HIDDEN_LAYERS']
        ),
    }

    trunk_config = {
        'architecture': trunk_architecture,
        'layers': (
            [trunk_input_size]
            + model_params['TRUNK_HIDDEN_LAYERS']
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

    output_handling = model_params.get('OUTPUT_HANDLING', 'single_trunk_single_branch').lower()
    output_strategy_mapping = {
        'single_trunk_split_branch': SingleTrunkSplitBranchStrategy,
        'split_trunk_single_branch': SplitTrunkSingleBranchStrategy,
        'multiple_trunks_single_branch': MultipleTrunksSingleBranchStrategy,
        'single_trunk_multiple_branches': SingleTrunkMultipleBranchesStrategy,
        'multiple_trunks_multiple_branches': MultipleTrunksMultipleBranchesStrategy,
    }

    if output_handling not in output_strategy_mapping:
        raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")

    output_strategy = output_strategy_mapping[output_handling]()

    model_params['BASIS_CONFIG'] = output_strategy.get_basis_config()['type']

    training_strategy_name = model_params.get('TRAINING_STRATEGY').lower()

    if training_strategy_name == 'pod':
        inference_mode = kwargs.get('inference', False)
        training_strategy = PODTrainingStrategy(data=data, var_share=var_share, inference=inference_mode)

    elif training_strategy_name == 'two_step':
        train_dataset_length = len(data['xb']) if data else None
        if train_dataset_length is None:
            training_strategy = TwoStepTrainingStrategy()
        else:
            training_strategy = TwoStepTrainingStrategy(train_dataset_length=train_dataset_length)

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
        n_basis_functions=model_params['BASIS_FUNCTIONS']
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
    if model_params.get('OUTPUT_HANDLING', False):
        if 'single_trunk' in model_params.get('OUTPUT_HANDLING').lower():
            model_name += '_' + 'single' + '_'
            model_name += 'basis'
        elif 'multiple_trunks' in model_params.get('OUTPUT_HANDLING').lower():
            model_name += '_' + 'multiple' + '_'
            model_name += 'trunks'
        elif 'split_trunk' in model_params.get('OUTPUT_HANDLING').lower():
            model_name += '_' + 'multiple' + '_'
            model_name += 'basis'

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

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model, _ = create_model(
        model_params,
        inference=True
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    training_strategy = model_params.get('TRAINING_STRATEGY', '').lower()
    if training_strategy == 'two_step':
        model.training_strategy.set_matrices(Q_list=checkpoint.get('Q'),
                                                     R_list=checkpoint.get('R'),
                                                     T_list=checkpoint.get('T'))
    elif training_strategy == 'pod':
        model.training_strategy.set_basis(pod_basis=checkpoint.get('pod_basis'),
                                                     mean_functions=checkpoint.get('pod_basis'))
    
    model.training_strategy.inference_mode()
    model.eval()
    return model, model_params
