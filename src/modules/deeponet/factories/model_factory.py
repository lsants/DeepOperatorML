import os
import yaml
import torch
import warnings
from typing import Dict, Any, Tuple
from ..deeponet import DeepONet
from .activation_factory import ActivationFactory
from ...utilities.config_utils import process_config
from .loss_factory import LossFactory
from .optimizer_factory import OptimizerFactory
from ..factories.network_factory import NetworkFactory
from .strategy_factory import StrategyFactory

class ModelFactory:
    @staticmethod
    def create_model(model_params: Dict[str, Any], **kwargs) -> Tuple[DeepONet, str]:
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
        if branch_arch in ['mlp', 'resnet', 'cnn']:
            branch_config['activation'] = ActivationFactory.get_activation(model_params.get('BRANCH_ACTIVATION'))
        elif branch_arch == 'kan':
            branch_config['degree'] = model_params.get('BRANCH_DEGREE')

        trunk_arch = model_params['TRUNK_ARCHITECTURE'].lower()
        if trunk_arch in ['mlp', 'resnet', 'cnn']:
            trunk_config['activation'] = ActivationFactory.get_activation(model_params.get('TRUNK_ACTIVATION'))
        elif trunk_arch == 'kan':
            trunk_config['degree'] = model_params.get('TRUNK_DEGREE')

        trunk_config.setdefault("type", "trainable")
        branch_config.setdefault("type", "trainable")

        output_handling = StrategyFactory.get_output_handling(model_params['OUTPUT_HANDLING'], model_params['OUTPUT_KEYS'])
        model_params['BASIS_CONFIG'] = output_handling.BASIS_CONFIG

        loss_function = LossFactory.get_loss_function(model_params['LOSS_FUNCTION'], model_params)

        training_strategy = StrategyFactory.get_training_strategy(
            model_params.get('TRAINING_STRATEGY'),
            loss_function,
            data,
            model_params,
            inference=kwargs.get('inference', False)
        )

        model = DeepONet(
            base_branch_config=branch_config,
            base_trunk_config=trunk_config,
            output_handling=output_handling,
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
        training_strategy = model_params.get('TRAINING_STRATEGY', '').lower()

        checkpoint = torch.load(model_path, map_location=device)

        if training_strategy == 'two_step':
            model_params['TRAINED_TRUNK'] = checkpoint.get('trained_trunk')
            trained_trunk = checkpoint.get('trained_trunk')
            print("KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK", trained_trunk)
        elif training_strategy == 'pod':
            model.training_strategy.set_pod_data(
                pod_basis=checkpoint.get('pod_basis'),
                mean_functions=checkpoint.get('mean_functions')
            )

        model, _ = ModelFactory.create_model(model_params, inference=True, trained_trunk=trained_trunk)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)


        model.eval()
        return model, model_params