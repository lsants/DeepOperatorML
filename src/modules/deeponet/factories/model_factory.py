from __future__ import annotations
import os
import yaml
import torch
import logging
from typing import Any
from ..deeponet import DeepONet
from .loss_factory import LossFactory
from .strategy_factory import StrategyFactory
from .activation_factory import ActivationFactory
from ...utilities.config_utils import process_config
from ...data_processing.transforms import Compose, Rescale

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_model(model_params: dict[str, Any], **kwargs) -> tuple[DeepONet, str]:
        data = kwargs.get('train_data')
        inference = kwargs.get('inference', False)

        trunk_input_size = len(model_params['COORDINATE_KEYS'])
        if model_params.get('TRUNK_FEATURE_EXPANSION', 0) > 0:
            trunk_input_size += 2 * \
                len(model_params['COORDINATE_KEYS']) * \
                model_params['TRUNK_FEATURE_EXPANSION']

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
            branch_config['activation'] = ActivationFactory.get_activation(
                model_params.get('BRANCH_ACTIVATION'))
        elif branch_arch == 'kan':
            branch_config['degree'] = model_params.get('BRANCH_DEGREE')

        trunk_arch = model_params['TRUNK_ARCHITECTURE'].lower()
        if trunk_arch in ['mlp', 'resnet', 'cnn']:
            trunk_config['activation'] = ActivationFactory.get_activation(
                model_params.get('TRUNK_ACTIVATION'))
        elif trunk_arch == 'kan':
            trunk_config['degree'] = model_params.get('TRUNK_DEGREE')

        trunk_config.setdefault("type", "trainable")
        branch_config.setdefault("type", "trainable")

        output_handling = StrategyFactory.get_output_handling(
            model_params['OUTPUT_HANDLING'], model_params['OUTPUT_KEYS'])
        model_params['BASIS_CONFIG'] = output_handling.BASIS_CONFIG

        loss_function = LossFactory.get_loss_function(
            model_params['LOSS_FUNCTION'], model_params)
        transforms = Compose([
            Rescale(factor=model_params["BASIS_FUNCTIONS"],
                    config=model_params["RESCALING"])
        ])

        training_strategy = StrategyFactory.get_training_strategy(
            strategy_name=model_params.get('TRAINING_STRATEGY'),
            loss_fn=loss_function,
            data=data,
            model_params=model_params,
            transform=transforms,
            inference=inference,
            trained_trunk=kwargs.get('trained_trunk'),
            pod_trunk=kwargs.get('pod_trunk'),
        )

        model = DeepONet(
            branch_config=branch_config,
            trunk_config=trunk_config,
            output_handling=output_handling,
            training_strategy=training_strategy,
            n_outputs=len(model_params['OUTPUT_KEYS']),
            n_basis_functions=model_params['BASIS_FUNCTIONS']
        ).to(device=model_params['DEVICE'], dtype=getattr(torch, model_params['PRECISION']))

        model_params["BASIS_FUNCTIONS"] = model.n_basis_functions

        model_params = process_config(model_params)
        if 'MODEL_NAME' not in model_params or not model_params['MODEL_NAME']:
            raise ValueError("MODEL_NAME is missing in the configuration.")
        model_name = model_params['MODEL_NAME']

        return model, model_name

    @staticmethod
    def initialize_model(trained_model_config: dict[str, any], device, **kwargs) -> DeepONet:
        checkpoint_path = trained_model_config["CHECKPOINTS_PATH"]
        best_model_state_path = os.path.join(checkpoint_path, 'best_model_state.pth')

        checkpoint = torch.load(f=best_model_state_path, map_location=device, weights_only=True)

        if trained_model_config["TRAINING_STRATEGY"] == 'two_step':
            trained_trunk = checkpoint.get('trunk.trained_tensor')
            model, _ = ModelFactory.create_model(model_params=trained_model_config,
                                                 inference=True,
                                                 trained_trunk=trained_trunk
                                                 )
        elif trained_model_config["TRAINING_STRATEGY"] == 'pod':
            saved_pod_trunk = {'basis': checkpoint.get(
                'pod_basis'), 'mean': checkpoint.get('mean_functions')}
            model, _ = ModelFactory.create_model(model_params=trained_model_config,
                                                 inference=True,
                                                 pod_trunk=saved_pod_trunk
                                                 )
        else:
            model, _ = ModelFactory.create_model(model_params=trained_model_config,
                                                 inference=True)
        
        model.load_state_dict(state_dict=checkpoint, strict=False)
        model.eval()

        return model
