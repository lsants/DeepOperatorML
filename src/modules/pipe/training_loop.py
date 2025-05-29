from __future__ import annotations
import torch
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from ..data_processing.deeponet_sampler import DeepONetSampler
from .history import HistoryStorer
from ..model.training_strategies.base import TrainingStrategy
from ..model.deeponet import DeepONet

logger = logging.getLogger(__name__)


class TrainingLoop:
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
    ) -> None:
        # Core references --------------------------------------------------
        self.device: torch.device = torch.device(device)
        self.model: DeepONet = model.to(self.device)
        self.strategy: TrainingStrategy = strategy
        self.train_loader: DataLoader = train_loader
        self.val_loader: Optional[DataLoader] = val_loader
        self.sampler: DeepONetSampler = sampler

        # Let strategy freeze layers / register hooks / etc.
        self.strategy.setup_training(self.model)

        # Optimisation schedule [(n_epochs, optimiser, scheduler?), ...]
        self.optimizer_specs: List[
            Tuple[int, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]
        ] = self.strategy.get_train_schedule()
        if not self.optimizer_specs:
            raise ValueError("Strategy produced an empty training schedule.")

        # Phase bookkeeping -----------------------------------------------
        self.phases: List[str] = self.strategy.get_phases()
        self.current_phase: int = 1  # 1‑indexed for readability
        self.current_spec_idx: int = 0
        self.epochs_in_current_spec: int = 0
        self._init_current_optimizer()

        # History + checkpoints -------------------------------------------
        self.history = HistoryStorer(phases=self.phases)
        self.checkpoint_dir = Path(checkpoint_dir)

    # ------------------------------------------------------------------ private
    def _init_current_optimizer(self) -> None:
        epochs, opt, sch = self.optimizer_specs[self.current_spec_idx]
        self.epochs_per_spec: int = epochs
        self.optimizer: torch.optim.Optimizer = opt
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = sch

        # Ensure optimiser params reside on correct device
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
    def run(self, total_epochs: int) -> None:
        for epoch in range(1, total_epochs + 1):
            # — progression BEFORE epoch —
            self._handle_spec_progression()
            # ---------------- train ----------------
            logger.info(f"\n\n\n\n================= Epoch {epoch} ===================== \n\n\n\n")
            train_metrics = self._run_epoch(train=True)
            self._store_metrics(train_metrics, train=True)

            # ---------------- validate --------------
            val_metrics: Dict[str, float] = {}
            if self.strategy.validation_enabled() and self.val_loader is not None:
                val_metrics = self._run_epoch(train=False)
                self._store_metrics(val_metrics, train=False)

            # LR bookkeeping
            self.history.store_learning_rate(
                phase=self.phases[self.current_phase - 1],
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )

            # Strategy hook (early‑stopping, logging, …)
            if hasattr(self.strategy, "on_epoch_end"):
                self.strategy.on_epoch_end(  # type: ignore[attr-defined]
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    metrics={"train": train_metrics, "val": val_metrics},
                )

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging ------------------------------------------------------
            self._log_progress(epoch, train_metrics, val_metrics)

            # epoch counter for spec
            self.epochs_in_current_spec += 1

        exp_path = self.checkpoint_dir / 'experiment.pt'
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "strategy": self.strategy,
                "history": self.history.get_history(),
                "phase": self.current_phase,
            },
            exp_path,
        )

    # -------------------------------------------------------------- core epoch
    def _run_epoch(self, *, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        loader: DataLoader = self.train_loader if train else self.val_loader  # type: ignore[assignment]
        assert loader is not None, "Validation loader requested but not provided."

        aggregated: Dict[str, float] = defaultdict(float)
        processed_samples = 0  # will replace old bs*num_batches logic

        # Derive total sample count for later normalisation -----------------
        aggregated: Dict[str, float] = defaultdict(float)

        processed_samples = 0

        context = torch.enable_grad() if train else torch.inference_mode()
        try:
            with context:
                for i, batch in enumerate(loader):
                    x_branch, x_trunk, y_true = self._prepare_batch(batch)

                    # Forward + loss
                    y_pred, loss = self.strategy.compute_loss(
                        self.model, x_branch, x_trunk, y_true
                    )

                    # if i % 10 == 0:
                    #     print(f"Loss: {loss:.4E} for batch {i + 1}")
                    print(f"Loss: {loss:.4E} for batch {i + 1}")

                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.strategy.apply_gradient_constraints(self.model)
                        self.optimizer.step()

                    # Per-batch metrics ------------------------------------
                    batch_metrics = self.strategy.calculate_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        loss=loss.item(),
                        train=train,
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
                    "partial_epoch_metrics": partial_epoch_metrics,
                },
                self.checkpoint_dir / "INTERRUPTED.pt",
            )
            print(f"Checkpoint written to {self.checkpoint_dir/'INTERRUPTED.pt'}")
            raise  # bubble up so outer loop can exit cleanly

        # --------------- epoch-level normalisation -------------------------

        return {k: v / processed_samples for k, v in aggregated.items()}


    # ---------------------------------------------------------------- metrics
    def _store_metrics(self, metrics: Dict[str, float], *, train: bool) -> None:
        """Persist epoch‑level metrics inside :class:`HistoryStorer`."""
        if not metrics:
            return
        phase = self.phases[self.current_phase - 1]

        # Standard fields --------------------------------------------------
        if "loss" in metrics:
            if train:
                self.history.store_epoch_train_loss(phase, metrics["loss"])
            else:
                self.history.store_epoch_val_loss(phase, metrics["loss"])
        if "error" in metrics:
            if train:
                self.history.store_epoch_train_errors(phase, {"error": metrics["error"]})
            else:
                self.history.store_epoch_val_errors(phase, {"error": metrics["error"]})

        # Any extra keys are treated as error‑like (extend as needed)
        for k, v in metrics.items():
            if k in {"loss", "error"}:
                continue
            if train:
                self.history.store_epoch_train_errors(phase, {k: v})
            else:
                self.history.store_epoch_val_errors(phase, {k: v})

    # ---------------------------------------------------------------- batch util
    def _prepare_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            xb, xt, y = batch["xb"], batch["xt"], batch["g_u"]
        else:
            xb, xt, y = batch  # type: ignore[misc]

        return (
            xb.to(self.device, non_blocking=True),
            xt.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

    # -------------------------------------------------------------- progression
    def _handle_spec_progression(self) -> None:
        if self.epochs_in_current_spec < self.epochs_per_spec:
            return

        # Move to next spec within same phase
        if self.current_spec_idx < len(self.optimizer_specs) - 1:
            self.current_spec_idx += 1
            self.epochs_in_current_spec = 0
            self._init_current_optimizer()
            return

        # Otherwise trigger phase transition
        self._handle_phase_transition()

    def _handle_phase_transition(self) -> None:
        prev_phase_name = self.phases[self.current_phase - 1]
        logger.info("Phase '%s' completed. Transitioning …", prev_phase_name)

        # Strategy‑specific transition (may mutate model)
        self.strategy.execute_phase_transition(
            model=self.model,
            full_trunk_batch=self._get_full_trunk_batch()  # Provided for POD modes
        )

        # Prepare next phase ----------------------------------------------
        self.current_phase += 1
        if self.current_phase > len(self.phases):
            raise RuntimeError("All phases completed but training asked to continue.")
        self.history.add_phase(self.phases[self.current_phase - 1])

        # Refresh optimisation schedule as strategy may change it
        self.optimizer_specs = self.strategy.get_train_schedule()
        self.current_spec_idx = 0
        self.epochs_in_current_spec = 0
        self._init_current_optimizer()

        # Checkpoint
        self._save_checkpoint(f"phase_{prev_phase_name}_end.pt")

    # -------------------------------------------------------------- persistence
    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "strategy": self.strategy.state_dict(),
                "history": self.history.get_history(),
                "phase": self.current_phase,
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)

    # --------------------------------------------------------------- utilities
    def _log_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        msg = (
            f"[{self.strategy.get_phases()[self.current_phase - 1]} | Epoch {epoch}] "
            f"train_loss={train_metrics.get('loss', float('nan')):.4e} "
            f"train_err={train_metrics.get('error', float('nan')):.4e}"
        )
        if val_metrics:
            msg += (
                # f" | val_loss={val_metrics.get('loss', float('nan')):.4e} "
                f"val_err={val_metrics.get('error', float('nan')):.4e}"
            )
        logger.info(msg)

    def _get_full_trunk_batch(self) -> torch.Tensor:
        xs = []
        for _, x_trunk, _ in self.train_loader:  # type: ignore[misc]
            xs.append(x_trunk)
        return torch.cat(xs, dim=0).to(self.device)

    # ------------------------------------------------------------------- public
    def get_history(self) -> Dict[str, Dict[str, list]]:
        """Return accumulated training/validation history."""
        return self.history.get_history()
