import torch
import logging
from typing import TYPE_CHECKING, Optional, Callable
from .training_strategy_base import TrainingStrategy
from .helpers import DecompositionHelper, PhaseManager, TwoStepHelper
from ..components import PretrainedTrunk, TwoStepTrunk

if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)


class TwoStepTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn: callable, inference: bool = False, output_transform: Optional[Callable] = None, **kwargs):
        super().__init__(loss_fn, output_transform)
        self.inference = inference
        self.phase_manager = PhaseManager()
        self.decomposition_helper = DecompositionHelper()
        self.two_step_helper = TwoStepHelper(
            self.decomposition_helper, kwargs['device'], kwargs['precision'])
        self.A_dims = kwargs.get('train_dataset_length'),
        self.A = None
        self.current_phase = self.phase_manager.current_phase  # initially "trunk"
        self.pretrained_trunk_tensor = kwargs.get(
            "pretrained_trunk_tensor", None)
        if self.inference:
            self.current_phase = "branch"
            logger.info(
                "TwoStepTrainingStrategy: Initialized in inference mode.")

    def prepare_training(self, model: "DeepONet", **kwargs) -> None:
        if self.inference:
            logger.info(
                "TwoStepTrainingStrategy (inference): No training preparation required.")
            return
        if self.A is None:
            branch_output_size = (model.output_handling.branch_output_size,)
            A_dims = self.A_dims + branch_output_size
            self.A = self.two_step_helper.set_A_matrix(*A_dims)
            model.trunk = TwoStepTrunk(model.trunk, self.A)

    def forward(self, model: "DeepONet", xb: torch.Tensor | None = None, xt: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor]:
        branch_out, trunk_out = self.two_step_helper.compute_outputs(
            model, xb, xt, self.current_phase)
        if not self.inference and self.current_phase == 'branch':
            # If there are multiple branches (cols of A = n * number of basis functions), break A into a tuple containing n matrices
            if branch_out.shape[1] > model.n_basis_functions:
                slice_size = branch_out.shape[1] // model.n_outputs
                slices = [
                    branch_out[:, i * slice_size: (i + 1) * slice_size] for i in range(model.n_outputs)]
                outputs = tuple(slices)
            else:
                outputs = tuple(branch_out for _ in range(model.n_outputs))
            return outputs
        return model.output_handling.forward(model, branch_out, trunk_out)

    def compute_loss(self, outputs: tuple, batch: dict[str, torch.Tensor], model: "DeepONet", params: dict, **kwargs) -> float:
        return self.two_step_helper.compute_loss(outputs, batch, model, params, self.current_phase, self.loss_fn)

    def compute_errors(self, outputs: tuple, batch: dict[str, torch.Tensor], model: "DeepONet", params: dict, **kwargs) -> dict[str, float]:
        return self.two_step_helper.compute_errors(outputs, batch, model, params, self.current_phase)

    def get_trunk_config(self, trunk_config: dict) -> dict:
        config = trunk_config.copy()
        if self.inference or self.current_phase == "branch":
            config["type"] = "pretrained"
            if self.pretrained_trunk_tensor is None:
                raise ValueError(
                    "TwoStepTrainingStrategy: Pretrained trunk tensor should be available in inference mode.")
            config["fixed_tensor"] = self.pretrained_trunk_tensor
        else:
            config["type"] = "trainable"
        return config

    def get_branch_config(self, branch_config: dict) -> dict:
        config = branch_config.copy()
        config["type"] = "trainable"
        return config

    def update_training_phase(self, phase: str) -> None:
        self.phase_manager.update_phase(phase)
        self.current_phase = self.phase_manager.current_phase
        logger.info(
            f"TwoStepTrainingStrategy: Updated phase to {self.current_phase}")

    def prepare_for_phase(self, model: "DeepONet", **kwargs) -> None:
        """
        In branch phase, compute the pretrained trunk tensor and update the model's trunk component.
        """
        if self.inference:
            logger.info(
                "TwoStepTrainingStrategy (inference): No phase preparation needed.")
            return

        self.phase_manager.prepare_phase(model)

        if self.current_phase == "branch":
            trunk_input = kwargs.get("train_batch")['xt']
            if trunk_input is None:
                raise ValueError(
                    "TwoStepTrainingStrategy: Missing trunk input for phase transition.")
            decomposition_params = kwargs.get('model_params')
            self.pretrained_trunk_tensor = self.two_step_helper.compute_trained_trunk(
                model, decomposition_params, trunk_input
            )
            logger.info(
                "TwoStepTrainingStrategy: Pretrained trunk tensor computed via helper.")
            model.trunk = PretrainedTrunk(self.pretrained_trunk_tensor)
            logger.info(
                "TwoStepTrainingStrategy: Model trunk updated to PretrainedTrunk.")

    def after_epoch(self, epoch: int, model: "DeepONet", params: dict[str, any], **kwargs) -> None:
        self.two_step_helper.after_epoch(epoch, model, params, **kwargs)

    def get_phases(self) -> list[str]:
        return ['trunk', 'branch']
