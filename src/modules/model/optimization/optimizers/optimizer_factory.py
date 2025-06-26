from __future__ import annotations
import torch
from .optimizers import OPTIMIZER_MAP, SCHEDULER_MAP
from typing import TYPE_CHECKING
from .config import OptimizerSpec


def create_optimizer(spec: 'OptimizerSpec', params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    optimizer_class = OPTIMIZER_MAP[spec.optimizer_type.lower()]
    optimizer = optimizer_class(
        params,
        lr=spec.learning_rate,
        weight_decay=spec.l2_regularization
    )
    return optimizer


def create_scheduler(spec: 'OptimizerSpec', optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    if spec.lr_scheduler is None:
        return None
    scheduler_type = spec.lr_scheduler.get("type", "step")
    if scheduler_type not in SCHEDULER_MAP:
        raise ValueError(f"Unsupported scheduler {scheduler_type}")
    scheduler_class = SCHEDULER_MAP[scheduler_type]

    return scheduler_class(
        optimizer,
        **{k: v for k, v in spec.lr_scheduler.items() if k != "type"}
    )
