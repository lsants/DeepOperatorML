from __future__ import annotations
import torch
import logging
from collections import defaultdict
from pathlib import Path
from dataclasses import asdict
from typing import Iterable, Optional
from torch.utils.data import DataLoader
from src.modules.models.deeponet.dataset.deeponet_sampler import DeepONetSampler
from src.modules.pipe.history import HistoryStorer
from src.modules.models.deeponet.training_strategies.base import TrainingStrategy
from src.modules.models.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)


class DeepONetTrainingLoop:
    """Strategy‑aware training driver for *DeepONet*.

    The loop itself is deliberately thin: it orchestrates epochs, delegates
    all model‑specific behaviour to the provided ``TrainingStrategy`` and keeps
    a clean metric history through ``HistoryStorer``.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(
        self,
        model: DeepONet,
        strategy: TrainingStrategy,
        train_loader: DataLoader,
        sampler: DeepONetSampler,
        checkpoint_dir: str | Path,
        val_loader: Optional[DataLoader] = None,
        device: str | torch.device = "cpu",
        label_map: list[str] | None = None
    ) -> None:
        # Core references --------------------------------------------------
        self.device: torch.device = torch.device(device)
        self.model: DeepONet = model.to(self.device)
        self.strategy: TrainingStrategy = strategy
        self.train_loader: DataLoader = train_loader
        self.val_loader: Optional[DataLoader] = val_loader
        self.sampler: DeepONetSampler = sampler

        self.strategy.setup_training(self.model)

        self.optimizer_specs: list[
            tuple[int, torch.optim.optimizer.Optimizer,
                  Optional[torch.optim.lr_scheduler._LRScheduler]]
        ] = self.strategy.get_train_schedule()
        if not self.optimizer_specs:
            raise ValueError("Strategy produced an empty training schedule.")

        # Phase bookkeeping -----------------------------------------------
        self.phases: list[str] = self.strategy.get_phases()
        self.current_phase: int = 1  # 1‑indexed for readability
        self.current_spec_idx: int = 0
        self.epochs_in_current_spec: int = 0
        self._init_current_optimizer()

        # History + checkpoints -------------------------------------------
        self.history = HistoryStorer(phases=self.phases)
        self.checkpoint_dir = Path(checkpoint_dir)

        self.strategy_dict = asdict(self.strategy.config)
        if self.strategy.config.name == 'two_step':
            self.strategy_dict.update({
                'trunk_inner_config': self.strategy._original_trunk_cfg,  # type: ignore[attr-defined]
                'branch_inner_config': self.strategy._original_branch_cfg,  # type: ignore[attr-defined]
            })

        # Label map -------------------------------------------------------
        self.label_map = label_map

    # ------------------------------------------------------------------ private

    def _init_current_optimizer(self) -> None:
        epochs, opt, sch = self.optimizer_specs[self.current_spec_idx]
        self.epochs_per_spec: int = epochs

        self.optimizer: torch.optim.optimizer.Optimizer = opt
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = sch

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.data = p.data.to(self.device)
                if p.grad is not None:
                    p.grad = p.grad.to(self.device)

        logger.debug(
            "Switched to optimiser spec %d | epochs=%d | lr=%g",
            self.current_spec_idx,
            self.epochs_per_spec,
            self.optimizer.param_groups[0]["lr"],
        )

    # ----------------------------------------------------------------- training
    def run(self) -> None:
        epoch = 0
        while True:
            epoch += 1
            if self.epochs_in_current_spec >= self.epochs_per_spec:
                if self.current_spec_idx < len(self.optimizer_specs) - 1:
                    self.current_spec_idx += 1
                    self.epochs_in_current_spec = 0
                    self._init_current_optimizer()

                elif self.strategy.should_transition_phase(current_phase=self.current_phase, current_epoch=epoch):
                    self._handle_phase_transition()
                    epoch = 0
                else:
                    break

            # ---------------- train ----------------
            # logger.info(
            #     f"\n\n\n\n================= Epoch {epoch} ===================== \n\n\n\n")

            train_metrics = self._run_epoch(train=True)
            self.history.store_epoch_metrics(
                phase=self.phases[self.current_phase - 1],
                loss=train_metrics.get("loss"),
                errors={k: v for k, v in train_metrics.items() if k != "loss"},
                train=True
            )
            max_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    max_gradients[name] = param.grad.max().item()
                else:
                    max_gradients[name] = None

            self.history.store_max_gradients(
                phase=self.phases[self.current_phase - 1],
                gradients=max_gradients
            )

            # ---------------- validate --------------
            val_metrics: dict[str, float] = {}
            if self.strategy.validation_enabled() and self.val_loader is not None:
                val_metrics = self._run_epoch(train=False)
                self.history.store_epoch_metrics(
                    phase=self.phases[self.current_phase - 1],
                    loss=val_metrics.get("loss"),
                    errors={k: v for k, v in val_metrics.items() if k !=
                            "loss"},
                    train=False
                )

            self.history.store_learning_rate(
                phase=self.phases[self.current_phase - 1],
                lr=self.optimizer.param_groups[0]["lr"],
            )

            if hasattr(self.strategy, "on_epoch_end"):
                self.strategy.on_epoch_end(  # type: ignore[attr-defined]
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    metrics={"train": train_metrics, "val": val_metrics},
                )

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging ------------------------------------------------------
            self._log_progress(epoch, train_metrics, val_metrics)

            self.epochs_in_current_spec += 1

        exp_path = self.checkpoint_dir / 'experiment.pt'
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "history": self.history.get_history(),
                "phase": self.current_phase,
                "strategy": self.strategy_dict,
            },
            exp_path,
        )

    # -------------------------------------------------------------- core epoch
    def _run_epoch(self, *, train: bool) -> dict[str, float]:
        self.model.train(mode=train)
        loader: DataLoader = self.train_loader if train else self.val_loader # type: ignore[assignment]
        assert loader is not None, "Validation loader requested but not provided."

        aggregated: dict[str, float] = defaultdict(float)
        processed_samples = 0

        context = torch.enable_grad() if train else torch.inference_mode()
        try:
            with context:
                for i, batch in enumerate(loader):
                    x_branch, x_trunk, y_true, indices = self._prepare_batch(batch)

                    y_pred, loss = self.strategy.compute_loss(
                        model=self.model, 
                        x_branch=x_branch, 
                        x_trunk=x_trunk, 
                        y_true=y_true,
                        indices=indices
                    )

                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.strategy.apply_gradient_constraints(self.model)
                        self.optimizer.step()
                        
                    # Per-batch metrics ------------------------------------
                    batch_metrics = self.strategy.calculate_metrics(
                        model=self.model,
                        y_true=y_true,
                        y_pred=y_pred,
                        loss=loss.item(),
                        train=train,
                        branch_indices=indices[0],
                        label_map=self.label_map
                    )

                    bs = y_true.size(0)

                    processed_samples += bs
                    for k, v in batch_metrics.items():
                        aggregated[k] += v * bs

        # --------------- Ctrl-C handling --------------------------
        except KeyboardInterrupt:
            print("\nInterrupted by user – saving checkpoint …")
            # Save *partial* metrics inside checkpoint for later inspection
            partial_epoch_metrics = {
                k: (v / processed_samples if processed_samples else float("nan"))
                for k, v in aggregated.items()
            }

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                    "history": self.history.get_history(),
                    "strategy": self.strategy_dict,  #
                    "partial_epoch_metrics": partial_epoch_metrics,
                },
                self.checkpoint_dir / "INTERRUPTED.pt",
            )
            print(
                f"Checkpoint written to {self.checkpoint_dir/'INTERRUPTED.pt'}")
            raise  # bubble up so outer loop can exit cleanly

        # --------------- epoch-level normalisation -------------------------

        return {k: v / processed_samples for k, v in aggregated.items()}

    # ---------------------------------------------------------------- batch util

    def _prepare_batch(
        self,
        batch: dict[str, torch.Tensor | tuple[Iterable[int], ...]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[Iterable[int], ...]]:
        if isinstance(batch, dict):
            xb, xt, y, idx = batch["xb"], batch["xt"], batch["g_u"], batch["indices"]
        else:
            raise TypeError(
                f"Expected batch to be a dict, got {type(batch)} instead."
            )
        if not isinstance(xb, torch.Tensor) or not isinstance(xt, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "Batch tensors must be of type torch.Tensor, got "
                f"{type(xb)}, {type(xt)}, {type(y)} instead."
            )
        if not isinstance(idx, tuple):
            raise TypeError(
                "Index must be a tuple of indices, got "
                f"{type(idx)} instead."
            )
        return (
            xb.to(self.device, non_blocking=True),
            xt.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
            idx
        )

    def _handle_phase_transition(self) -> None:
        prev_phase_name = self.phases[self.current_phase - 1]
        logger.info("Phase '%s' completed. Transitioning …", prev_phase_name)

        self.strategy.execute_phase_transition(
            model=self.model,
            all_branch_indices=self._get_full_branch_batch(),
            full_trunk_batch=self._get_full_trunk_batch(),
            full_outputs_batch=self._get_full_outputs_batch()
        )
        self.current_phase += 1
        if self.current_phase > len(self.phases):
            raise RuntimeError(
                "All phases completed but training asked to continue.")
        self.history.add_phase(self.phases[self.current_phase - 1])
        self.optimizer_specs = self.strategy.get_train_schedule()
        self.current_spec_idx = 0
        self.epochs_in_current_spec = 0

        self._init_current_optimizer()

        self._save_checkpoint(f"phase_{prev_phase_name}_end.pt")

    # -------------------------------------------------------------- persistence
    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "history": self.history.get_history(),
                "strategy": self.strategy_dict,
                "phase": self.current_phase,
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)

    # --------------------------------------------------------------- utilities

    def _log_progress(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        msg = (
            f"[{self.strategy.get_phases()[self.current_phase - 1]} | Epoch {epoch}] \n"
            f"train_loss={train_metrics.get('loss', float('nan')):.4e} \n"
        )

        train_error_parts = []
        for key in (train_metrics.keys()):
            if key.startswith('Error'):
                train_error_parts.append(
                    f"train_{key}={train_metrics[key]:.4e}\n")

        if train_error_parts:
            msg += " ".join(train_error_parts)

        if val_metrics:
            msg += (
                f" | val_loss={val_metrics.get('loss', float('nan')):.4e} \n"
            )

            val_error_parts = []
            for key in (val_metrics.keys()):
                if key.startswith('Error'):
                    val_error_parts.append(
                        f"val_{key}={val_metrics[key]:.4e}\n")

            if val_error_parts:
                msg += " ".join(val_error_parts)

        logger.info(msg)

    def _get_full_trunk_batch(self) -> torch.Tensor: # Trunk is not batched
        for batch in self.train_loader:  # type: ignore[misc]
            xs = batch['xt']
            break
        return xs

    def _get_full_branch_batch(self) -> torch.Tensor:
        xs = []
        for batch in self.train_loader:  # type: ignore[misc]
            branch_indices = batch['indices'][0]
            xs.append(torch.tensor(branch_indices, dtype=torch.int))
        return torch.cat(xs, dim=0).to(self.device)
    
    def _get_full_outputs_batch(self) -> torch.Tensor:
        ys = []
        for batch in self.train_loader:  # type: ignore[misc]
            targets = batch['g_u']
            ys.append(targets)
        return torch.cat(ys, dim=0).to(self.device)

    # ------------------------------------------------------------------- public
    def get_history(self) -> dict[str, dict[str, list]]:
        """Return accumulated training/validation history."""
        return self.history.get_history()
