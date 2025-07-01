from __future__ import annotations
from collections import defaultdict


class HistoryStorer:
    def __init__(self, phases: list[str]) -> None:
        self.history: dict[str, dict] = {
            phase: {
                'train_loss':     [],
                'val_loss':       [],
                'train_errors':   defaultdict(list),
                'val_errors':     defaultdict(list),
                'learning_rate':  []
            }
            for phase in phases
        }

    def add_phase(self, phase: str):
        if phase not in self.history:
            self.history[phase] = {
                'train_loss':     [],
                'val_loss':       [],
                'train_errors':   defaultdict(list),
                'val_errors':     defaultdict(list),
                'learning_rate':  []
            }

    def store_learning_rate(self, phase: str, lr: float) -> None:
        self.history[phase]['learning_rate'].append(lr)

    def store_epoch_metrics(
        self, phase: str, *,
        loss: float | None = None,
        errors: dict[str, float] | None = None,
        train: bool
    ):
        h = self.history[phase]
        if loss is not None:
            target_loss = 'train_loss' if train else 'val_loss'
            h[target_loss].append(loss)

        if errors is not None:
            target_err = 'train_errors' if train else 'val_errors'
            for key, val in errors.items():
                h[target_err][key].append(val)

    def get_history(self) -> dict[str, dict]:
        return self.history
