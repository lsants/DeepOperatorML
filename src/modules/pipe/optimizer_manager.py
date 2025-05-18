# File: src/modules/deeponet/helpers/optimizer_scheduler_manager.py
from __future__ import annotations
import torch
import logging
from typing import Any
from ..deeponet.factories.optimizer_factory import OptimizerFactory

logger = logging.getLogger(__name__)

class OptimizerSchedulerManager:
    def __init__(self, model_config: dict[str, Any], model: torch.nn.Module) -> None:
        """
        Initializes the manager based on the configuration.
        The configuration can specify a global schedule or a phase-specific schedule.
        """
        training_strategy = model_config["TRAINING_STRATEGY"]
        self.model = model
        self.is_phase_specific = False
        self.optimizer_schedule = None

        if training_strategy == "two_step":
            self.is_phase_specific = True
            schedule = model_config.get("PHASE_OPTIMIZER_SCHEDULE", {})
            self.optimizer_schedule = {}
            for phase, schedule_list in schedule.items():
                self.optimizer_schedule[phase] = []
                for item in schedule_list:
                    optimizer_params = {
                        "LEARNING_RATE": item.get("LEARNING_RATE"),
                        "L2_REGULARIZATION": item.get("L2_REGULARIZATION", 0)
                    }
                    optimizer = OptimizerFactory.get_optimizer(
                        optimizer_name=item.get("OPTIMIZER", "adam"),
                        parameters=list(model.parameters()),
                        config=optimizer_params
                    )
                    scheduler = None
                    if "LR_SCHEDULER" in item and item["LR_SCHEDULER"]:
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer=optimizer,
                            step_size=item["LR_SCHEDULER"].get("STEP_SIZE"),
                            gamma=item["LR_SCHEDULER"].get("GAMMA")
                        )
                    self.optimizer_schedule[phase].append({
                        "EPOCHS": item["EPOCHS"],
                        "OPTIMIZER": optimizer,
                        "SCHEDULER": scheduler
                    })
            logger.info("OptimizerSchedulerManager: Phase-specific optimizer schedule configured.")
        else:
            schedule = model_config.get("GLOBAL_OPTIMIZER_SCHEDULE", None)
            self.optimizer_schedule = []
            if schedule is not None:
                for item in schedule:
                    optimizer_params = {
                        "LEARNING_RATE": item.get("LEARNING_RATE"),
                        "L2_REGULARIZATION": item.get("L2_REGULARIZATION", 0)
                    }
                    optimizer = OptimizerFactory.get_optimizer(
                        optimizer_name=item.get("OPTIMIZER", "adam"),
                        parameters=list(model.parameters()),
                        config=optimizer_params
                    )
                    scheduler = None
                    if "LR_SCHEDULER" in item and item["LR_SCHEDULER"]:
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer=optimizer,
                            step_size=item["LR_SCHEDULER"].get("STEP_SIZE"),
                            gamma=item["LR_SCHEDULER"].get("GAMMA")
                        )
                    self.optimizer_schedule.append({
                        "EPOCHS": item["EPOCHS"],
                        "OPTIMIZER": optimizer,
                        "SCHEDULER": scheduler
                    })
            else:
                optimizer_params = {
                    "LEARNING_RATE": model_config.get("LEARNING_RATE"),
                    "L2_REGULARIZATION": model_config.get("L2_REGULARIZATION", 0)
                }
                optimizer = OptimizerFactory.get_optimizer(
                    optimizer_name=model_config.get("OPTIMIZER", "adam"),
                    parameters=list(model.parameters()),
                    config=optimizer_params
                )
                scheduler = None
                if model_config.get("LR_SCHEDULING", False):
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=model_config.get("SCHEDULER_STEP_SIZE"),
                        gamma=model_config.get("SCHEDULER_GAMMA")
                    )
                self.optimizer_schedule = [{
                    "EPOCHS": float("inf"),
                    "OPTIMIZER": optimizer,
                    "SCHEDULER": scheduler
                }]
            logger.info("OptimizerSchedulerManager: Global optimizer schedule configured.")

    def get_active_optimizer(self, current_epoch: int, phase: str | None = None) -> dict[str, torch.optim.Optimizer]:
        if self.is_phase_specific:
            if phase is None:
                raise ValueError("Phase must be provided for phase-specific optimizer schedule.")
            schedule_list = self.optimizer_schedule.get(phase, None)
            if schedule_list is None:
                raise ValueError(f"No optimizer schedule found for phase '{phase}'.")
            for entry in schedule_list:
                if current_epoch < entry["EPOCHS"]:
                    return {"active": entry["OPTIMIZER"]}
            return {"active": schedule_list[-1]["OPTIMIZER"]}
        else:
            for entry in self.optimizer_schedule:
                if current_epoch < entry["EPOCHS"]:
                    return {"active": entry["OPTIMIZER"]}
            return {"active": self.optimizer_schedule[-1]["OPTIMIZER"]}

    def get_active_scheduler(self, current_epoch: int, phase: str | None = None) -> dict[str, Any]:
        if self.is_phase_specific:
            if phase is None:
                raise ValueError("Phase must be provided for phase-specific optimizer schedule.")
            schedule_list = self.optimizer_schedule.get(phase, None)
            if schedule_list is None:
                raise ValueError(f"No optimizer schedule found for phase '{phase}'.")
            for entry in schedule_list:
                if current_epoch < entry["EPOCHS"]:
                    return {"active": entry["SCHEDULER"]}
            return {"active": schedule_list[-1]["SCHEDULER"]}
        else:
            for entry in self.optimizer_schedule:
                if current_epoch < entry["EPOCHS"]:
                    return {"active": entry["SCHEDULER"]}
            return {"active": self.optimizer_schedule[-1]["SCHEDULER"]}

    def step_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        if scheduler is not None:
            scheduler.step()

