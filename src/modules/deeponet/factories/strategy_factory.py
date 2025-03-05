import torch
import warnings
from ..training_strategies import (
    TrainingStrategy,
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)
from ..output_handling import (
    SingleOutputHandling,
    ShareBranchHandling,
    ShareTrunkHandling,
    SplitOutputsHandling,
    OutputHandling
)

class StrategyFactory:
    @staticmethod
    def get_output_handling(strategy_name: str, output_keys: list) -> OutputHandling:
        strategy_name_lower = strategy_name.lower()
        output_stategy_mapping = {
                'single_output': SingleOutputHandling,
                'share_branch': ShareBranchHandling,
                'share_trunk': ShareTrunkHandling,
                'split_outputs': SplitOutputsHandling,
        }
        if strategy_name_lower not in output_stategy_mapping:
            raise ValueError(f"Unsupported OUTPUT_HANDLING strategy: {strategy_name_lower}.")
        if strategy_name_lower == 'single_output' and len(output_keys) != 1:
            raise ValueError(f"Invalid output handling, can't use {strategy_name_lower} strategy for a multi-output model.")
        elif len(output_keys) == 1:
            warnings.warn(
                f"Warning: There's little use in using a strategy for handling multiple outputs when the model has {len(output_keys)} outputs. Resources may be wasted."
            )
        return output_stategy_mapping[strategy_name_lower]()
    
    @staticmethod
    def get_training_strategy(strategy_name: str, loss_fn: callable, data: dict[str, torch.Tensor], model_params: dict[str, any], inference: bool, **kwargs) -> TrainingStrategy:
        strategy_name_lower = strategy_name.lower()
        if strategy_name_lower == 'pod':
            var_share = model_params.get('VAR_SHARE')
            if not inference:
                return PODTrainingStrategy(loss_fn=loss_fn, 
                                        data=data, 
                                        var_share=var_share, 
                                        inference=inference)
        elif strategy_name_lower == 'two_step':
            train_dataset_length = len(data['xb']) if data and 'xb' in data else None
            if train_dataset_length is None:
                if not inference:
                    raise ValueError(f"Initializing a TwoStep model without informing batch size is only possible when doing inference.\nCheck what you're doing!")
                return TwoStepTrainingStrategy(
                    loss_fn=loss_fn,
                    device=model_params['DEVICE'],
                    precision=getattr(torch, model_params['PRECISION']),
                    inference=inference,
                    pretrained_trunk_tensor=kwargs.get('trained_trunk') 
                )
            else:
                return TwoStepTrainingStrategy(
                    loss_fn=loss_fn,
                    device=model_params['DEVICE'],
                    precision=getattr(torch, model_params['PRECISION']),
                    train_dataset_length=train_dataset_length,
                    inference=inference,
                )
        elif strategy_name_lower == 'standard':
            return StandardTrainingStrategy(loss_fn=loss_fn, 
                                            inference=inference)
        else:
            raise ValueError(f"Unsupported TRAINING_STRATEGY: {strategy_name_lower}.")