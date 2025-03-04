import os
import yaml
import torch
import warnings
from ..utilities.config_utils import process_config
from ..deeponet.deeponet import DeepONet
from .activation_factory import ActivationFactory
from .loss_factory import LossFactory
# from factories.optimizer_factory import OptimizerFactory
from .strategy_factory import StrategyFactory

class ModelFactory:
    @staticmethod
    def create_model(model_params: dict[str, any], **kwargs) -> tuple[DeepONet, str]:
        model_params = process_config(model_params)
        if 'MODELNAME' not in model_params or not model_params['MODELNAME']:
            raise ValueError("MODELNAME is missing in the configuration.")
        model_name = model_params['MODELNAME']
        data = kwargs.get('train_data')

        trunk_input_size = len(model_params['COORDINATE_KEYS'])
        if model_params.get('TRUNK_FEATURE_EXPANSION', False):
            trunk_input_size += 2 * len(model_params['COORDINATE_KEYS']) * model_params['TRUNK_EXPANSION_FEATURES_NUMBER']

        branch_config = {
            'architecture': model_params['BRANCH_ARCHITECTURE'],
            'layers': [len(model_params['INPUT_FUNCTION_KEYS'])] + model_params['BRANCH_HIDDEN_LAYERS'],
        }
        trunk_config = {
            'architecture': model_params['TRUNK_ARCHITECTURE'],
            'layers': [trunk_input_size] + model_params['TRUNK_HIDDEN_LAYERS'],
        }

        branch_arch = model_params['BRANCH_ARCHITECTURE'].lower()
        trunk_arch = model_params['TRUNK_ARCHITECTURE'].lower()
        if branch_arch in ['mlp', 'resnet', 'cnn']:
            branch_config['activation'] = ActivationFactory.get_activation(model_params.get('BRANCH_ACTIVATION'))
        if trunk_arch in ['mlp', 'resnet', 'cnn']:
            trunk_config['activation'] = ActivationFactory.get_activation(model_params.get('TRUNK_ACTIVATION'))
        if branch_arch == 'kan':
            branch_config['degree'] = model_params.get('BRANCH_DEGREE')
        if trunk_arch == 'kan':
            trunk_config['degree'] = model_params.get('TRUNK_DEGREE')

        output_strategy = StrategyFactory.get_output_strategy(model_params['OUTPUT_HANDLING'], 
                                                              model_params['OUTPUT_KEYS']
                                                              )
        model_params['BASIS_CONFIG'] = output_strategy.BASIS_CONFIG

        loss_function = LossFactory.get_loss_function(model_params['LOSS_FUNCTION'],
                                                                   model_params
                                                                   )
        training_strategy = StrategyFactory.get_training_strategy(
            model_params.get('TRAINING_STRATEGY'),
            loss_function,
            data,
            model_params,
            inference=kwargs['inference']
        )

        model = DeepONet(
            branch_config=branch_config,
            trunk_config=trunk_config,
            output_strategy=output_strategy,
            training_strategy=training_strategy,
            n_outputs=len(model_params['OUTPUT_KEYS']),
            n_basis_functions=model_params['BASIS_FUNCTIONS']
        ).to(model_params['DEVICE'], dtype=getattr(torch, model_params['PRECISION']))

        return model, model_name
    
    @staticmethod
    def initialize_model(model_folder: str, model_name: str, device: str, precision: str) -> tuple[DeepONet, dict[str, any]]:
        model_path = os.path.join(model_folder, f"model_state_{model_name}.pth")
        config_path = os.path.join(model_folder, f"model_info_{model_name}.yaml")

        with open(config_path, 'r') as file:
            model_params = yaml.safe_load(file)

        model_params['DEVICE'] = device
        model_params['PRECISION'] = precision

        checkpoint = torch.load(model_path, map_location=device)

        model, _ = ModelFactory.create_model(model_params, inference=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        training_strategy = model_params.get('TRAINING_STRATEGY', '').lower()
        if training_strategy == 'two_step':
            model.training_strategy.set_matrices(
                Q=checkpoint.get('Q'),
                R=checkpoint.get('R'),
                T=checkpoint.get('T')
            )
        elif training_strategy == 'pod':
            model.training_strategy.set_pod_data(
                pod_basis=checkpoint.get('pod_basis'),
                mean_functions=checkpoint.get('mean_functions')
            )

        model.eval()
        return model, model_params