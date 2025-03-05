# File: src/modules/deeponet/helpers/optimizer_scheduler_manager.py
import torch
import logging
from typing import Dict, Any
from ...factories.optimizer_factory import OptimizerFactory

logger = logging.getLogger(__name__)

class OptimizerSchedulerManager:
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module):
        """
        Initializes the manager based on a schedule defined in the configuration.
        
        The config is expected to contain an "OPTIMIZER_SCHEDULE" key that is a list of dictionaries.
        Each dictionary should contain:
            - "epochs": integer, the epoch threshold (upper bound) for this optimizer setting.
            - "optimizer": string, the name of the optimizer (e.g. "adam").
            - Other optimizer parameters (like "LEARNING_RATE", "L2_REGULARIZATION", etc.)
            - Optionally, "lr_scheduler": dict with keys "step_size" and "gamma".
        
        If "OPTIMIZER_SCHEDULE" is not provided, a single optimizer is used.
        """
        self.schedule = config.get("OPTIMIZER_SCHEDULE", None)
        self.optimizer_schedule = []
        self.model = model 
        if self.schedule is not None:
            # Assume the schedule list is sorted by the "epochs" threshold.
            for item in self.schedule:
                optimizer_params = {
                    "LEARNING_RATE": item.get("LEARNING_RATE"),
                    "L2_REGULARIZATION": item.get("L2_REGULARIZATION", 0)
                }
                optimizer = OptimizerFactory.get_optimizer(
                    item.get("optimizer", "adam"),
                    list(model.parameters()),
                    optimizer_params
                )
                scheduler = None
                if "lr_scheduler" in item and item["lr_scheduler"]:
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=item["lr_scheduler"].get("step_size"),
                        gamma=item["lr_scheduler"].get("gamma")
                    )
                self.optimizer_schedule.append({
                    "EPOCHS": item["EPOCHS"],
                    "OPTIMIZER": optimizer,
                    "SCHEDULER": scheduler
                })
            logger.info("OptimizerSchedulerManager: Multiple optimizer schedule configured.")
        else:
            optimizer_params = {
                "LEARNING_RATE": config.get("LEARNING_RATE"),
                "L2_REGULARIZATION": config.get("L2_REGULARIZATION", 0)
            }
            optimizer = OptimizerFactory.get_optimizer(
                config.get("OPTIMIZER", "adam"),
                list(model.parameters()),
                optimizer_params
            )
            scheduler = None
            if config.get("LR_SCHEDULING", False):
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.get("SCHEDULER_STEP_SIZE"),
                    gamma=config.get("SCHEDULER_GAMMA")
                )
            self.optimizer_schedule.append({
                "epochs": float("inf"),
                "optimizer": optimizer,
                "scheduler": scheduler
            })
            logger.info("OptimizerSchedulerManager: Single optimizer configured.")

    def get_active_optimizer(self, current_epoch: int) -> Dict[str, torch.optim.Optimizer]:
        """
        Returns the optimizer that should be active at the current epoch.
        """
        for entry in self.optimizer_schedule:
            if current_epoch < entry["EPOCHS"]:
                return {"active": entry["OPTIMIZER"]}
        return {"active": self.optimizer_schedule[-1]["OPTIMIZER"]}

    def get_active_scheduler(self, current_epoch: int) -> Dict[str, Any]:
        """
        Returns the scheduler that should be active at the current epoch.
        """
        for entry in self.optimizer_schedule:
            if current_epoch < entry["EPOCHS"]:
                return {"active": entry["SCHEDULER"]}
        return {"active": self.optimizer_schedule[-1]["SCHEDULER"]}

    def step_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        if scheduler is not None:
            scheduler.step()
