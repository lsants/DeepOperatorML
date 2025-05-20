from __future__ import annotations
import torch
import warnings
from collections.abc import Callable, Iterable
from typing import Any
from . import (
    TrainingStrategy,
    StandardTrainingStrategy,
    TwoStepTrainingStrategy,
    PODTrainingStrategy
)
from ..output_handling import (
    SingleOutputHandling,
    ShareTrunkHandling,
    SplitOutputsHandling,
    OutputHandling
)

class StrategyFactory:
    @staticmethod
    def get_output_handling(strategy_name: str, num_outputs: int) -> OutputHandling:
        strategy_name_lower = strategy_name.lower()
        output_stategy_mapping = {
            'single_output': SingleOutputHandling,
            'share_trunk': ShareTrunkHandling,
            'split_outputs': SplitOutputsHandling,
        }
        if strategy_name_lower not in output_stategy_mapping:
            raise ValueError(
                f"Unsupported OUTPUT_HANDLING strategy: {strategy_name_lower}.")
        if strategy_name_lower == 'single_output' and num_outputs != 1:
            raise ValueError(
                f"Invalid output handling, can't use {strategy_name_lower} strategy for a multi-output model.")
        elif strategy_name_lower != 'single_output' and num_outputs == 1:
            warnings.warn(
                f"Warning: There's little use in using a strategy for handling multiple outputs when the model has {num_outputs} outputs. Resources may be wasted."
            )
        return output_stategy_mapping[strategy_name_lower]()

    @staticmethod
    def get_training_strategy(strategy_name: str, 
                              loss_fn: Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]], torch.Tensor], 
                              data: dict[str, torch.Tensor] | None, 
                              model_params: dict[str, Any],
                              inference: bool = False, 
                              **kwargs: Any) -> TrainingStrategy:

        strategy_name_lower = strategy_name.lower()
        if strategy_name_lower == 'pod':
            var_share = model_params.get('VAR_SHARE')
            if not inference:
                return PODTrainingStrategy(loss_fn=loss_fn,
                                           data=data,
                                           var_share=var_share,
                                           inference=inference,
                                           )
            else:
                return PODTrainingStrategy(loss_fn=loss_fn,
                                           data=data,
                                           var_share=var_share,
                                           inference=inference,
                                           pod_trunk=kwargs.get('pod_trunk'),
                                           )
        elif strategy_name_lower == 'two_step':
            branch_batch_size = kwargs.get('branch_batch_size')
            if branch_batch_size is None:
                if not inference:
                    raise ValueError(
                        f"Initializing a TwoStep model without informing branch batch size is only possible when doing inference.\nCheck what you're doing!")
                return TwoStepTrainingStrategy(
                    loss_fn=loss_fn,
                    device=model_params['DEVICE'],
                    precision=getattr(torch, model_params['PRECISION']),
                    pretrained_trunk_tensor=kwargs.get('trained_trunk'),
                    inference=inference,
                )
            else:
                return TwoStepTrainingStrategy(
                    loss_fn=loss_fn,
                    device=model_params['DEVICE'],
                    precision=getattr(torch, model_params['PRECISION']),
                    branch_batch_size=branch_batch_size,
                    inference=inference,
                )
        elif strategy_name_lower == 'standard':
            return StandardTrainingStrategy(loss_fn=loss_fn,
                                            inference=inference,                     
                                            )
        else:
            raise ValueError(
                f"Unsupported TRAINING_STRATEGY: {strategy_name_lower}.")
