import os
import warnings
import yaml
import torch
from ..utilities.config_utils import process_config
from ..deeponet.deeponet import DeepONet
from modules.deeponet.optimization.loss_fns import LOSS_FUNCTIONS
from modules.deeponet.nn.activation_fns import ACTIVATION_MAP
from ..deeponet.training_strategies import (
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)
from ..deeponet.output_strategies import (
    SingleOutputStrategy,
    ShareBranchStrategy,
    ShareTrunkStrategy,
    SplitNetworksStrategy,
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

        if name not in ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation function: '{name}'. Supported functions are: {list(ACTIVATION_MAP.keys())}")
        return ACTIVATION_MAP[name]

    def get_loss_function(name):
        """
        Maps a string name to the corresponding PyTorch loss function.

        Args:
            name (str): Name of the loss function (e.g., 'mse', 'mag-phase').

        Returns:
            function: Corresponding PyTorch loss function.

        Raises:
            ValueError: If the loss function name is not recognized.
        """
        if name not in LOSS_FUNCTIONS:
            raise ValueError(
                f"Unsupported loss function: '{name}'. Supported functions are: {list(LOSS_FUNCTIONS.keys())}")
        if name == "mag_phase":
            if len(model_params["OUTPUT_KEYS"]) != 2:
                raise ValueError(
                    f"Invalid loss function '{name}' for non-complex targets.")
        return LOSS_FUNCTIONS[name]

    model_params = process_config(model_params)

    if 'MODELNAME' not in model_params or not model_params['MODELNAME']:
        raise ValueError("MODELNAME is missing in the configuration.")
    model_name = model_params['MODELNAME']
    var_share = model_params.get('VAR_SHARE')
    data = kwargs.get('train_data')

    trunk_input_size = len(model_params['COORDINATE_KEYS'])
    if model_params.get('TRUNK_FEATURE_EXPANSION', False):
        trunk_input_size += (
            2 * len(model_params['COORDINATE_KEYS']) *
            model_params['TRUNK_EXPANSION_FEATURES_NUMBER']
        )

    branch_architecture = model_params['BRANCH_ARCHITECTURE']
    trunk_architecture = model_params['TRUNK_ARCHITECTURE']

    branch_config = {
        'architecture': branch_architecture,
        'layers': (
            [len(model_params['INPUT_FUNCTION_KEYS'])]
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
        branch_config['activation'] = get_activation_function(
            model_params.get('BRANCH_ACTIVATION'))

    if trunk_architecture.lower() == 'mlp':
        trunk_config['activation'] = get_activation_function(
            model_params.get('TRUNK_ACTIVATION'))

    if branch_architecture.lower() == 'resnet':
        branch_config['activation'] = get_activation_function(
            model_params.get('BRANCH_ACTIVATION'))

    if trunk_architecture.lower() == 'resnet':
        trunk_config['activation'] = get_activation_function(
            model_params.get('TRUNK_ACTIVATION'))

    if branch_architecture.lower() == 'kan':
        branch_config['degree'] = model_params.get('BRANCH_DEGREE')

    if trunk_architecture.lower() == 'kan':
        trunk_config['degree'] = model_params.get('TRUNK_DEGREE')

    output_handling = model_params['OUTPUT_HANDLING'].lower()
    output_strategy_mapping = {
        'single_output': SingleOutputStrategy,
        'share_branch': ShareBranchStrategy,
        'share_trunk': ShareTrunkStrategy,
        'split_networks': SplitNetworksStrategy,
    }

    if output_handling not in output_strategy_mapping:
        raise ValueError(
            f"Unsupported OUTPUT_HANDLING strategy: {output_handling}")
    elif output_handling == 'single_output' and len(model_params['OUTPUT_KEYS']) != 1:
        raise ValueError(
            f"Invalid output handling, can't use {output_handling} strategy for a model with {len(model_params['OUTPUT_KEYS'])} outputs")
    elif len(model_params['OUTPUT_KEYS']) == 1:
        warnings.warn(
            f"Warning.... There's little use in using a strategy for handling multiple outputs when the model has {len(model_params['OUTPUT_KEYS'])} output. Resources will be wasted.")

    output_strategy = output_strategy_mapping[output_handling]()

    model_params['BASIS_CONFIG'] = output_strategy.get_basis_config()['type']

    training_strategy_name = model_params.get('TRAINING_STRATEGY').lower()

    loss_function = get_loss_function(model_params["LOSS_FUNCTION"])

    if training_strategy_name == 'pod':
        inference_mode = kwargs.get('inference', False)
        training_strategy = PODTrainingStrategy(loss_fn=loss_function,
                                                data=data,
                                                var_share=var_share,
                                                inference=inference_mode)

    elif training_strategy_name == 'two_step':
        train_dataset_length = len(data['xb']) if data else None
        if train_dataset_length is None:
            training_strategy = TwoStepTrainingStrategy(loss_fn=loss_function, 
                                                        device=model_params['DEVICE'], 
                                                        precision=getattr(torch, model_params['PRECISION']),
                                                        )
        else:
            training_strategy = TwoStepTrainingStrategy(loss_fn=loss_function, 
                                                        device=model_params['DEVICE'],
                                                        precision=getattr(torch, model_params['PRECISION']),
                                                        train_dataset_length=train_dataset_length
                                                        )

    elif training_strategy_name == 'standard':
        training_strategy = StandardTrainingStrategy(loss_function)

    else:
        raise ValueError(
            f"Unsupported TRAINING_STRATEGY: {training_strategy_name}")

    model = DeepONet(
        branch_config=branch_config,
        trunk_config=trunk_config,
        output_strategy=output_strategy,
        training_strategy=training_strategy,
        n_outputs=len(model_params['OUTPUT_KEYS']),
        n_basis_functions=model_params['BASIS_FUNCTIONS']
    ).to(model_params['DEVICE'], dtype=getattr(torch, model_params['PRECISION']))

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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model, _ = create_model(
        model_params,
        inference=True
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    training_strategy = model_params.get('TRAINING_STRATEGY', '').lower()
    if training_strategy == 'two_step':
        model.training_strategy.set_matrices(Q=checkpoint.get('Q'),
                                             R=checkpoint.get('R'),
                                             T=checkpoint.get('T'))
    elif training_strategy == 'pod':
        model.training_strategy.set_basis(pod_basis=checkpoint.get('pod_basis'),
                                          mean_functions=checkpoint.get('mean_functions'))

    model.training_strategy.inference_mode()
    model.eval()
    return model, model_params
