from __future__ import annotations

"""Utility helpers to visualise *TrainingLoop* histories.

Both helpers are robust to missing validation data or multi‑output error
metrics (e.g. ``{"real": …, "imag": …}``).
"""

from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

error_label_map = {
    'vector_l2': r'$L_{2} [\%]$',
    'matrix_fro': r'$L_{F} [\%]$',
}

loss_label_map = {
    'mse': f'MSE',
    'huber': f'Huber'
}

# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def _pad(seq: List[Any], n: int, pad_val: Any = np.nan) -> List[Any]:
    """Right‑pad *seq* to length *n* with *pad_val*."""
    return seq + [pad_val] * (n - len(seq))


def align_epochs(raw_history: Dict[str, Dict[str, list]]) -> Dict[str, Dict[str, Any]]:
    """Convert raw ``HistoryStorer`` output into an epoch‑aligned structure.

    Returned dict (per phase)::
        {
            "epochs": [...],
            "train_loss": [...],
            "val_loss": [...],
            "learning_rate": [...],
            "train_errors": {key: [...]},
            "val_errors":   {key: [...]},
            "output_keys": [...],
        }
    """
    aligned: Dict[str, Dict[str, Any]] = {}

    for phase, metrics in raw_history.items():
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])
        lr_hist = metrics.get("learning_rate", [])
        train_err = metrics.get("train_errors", [])
        val_err = metrics.get("val_errors", [])

        # Collect all error keys that ever appeared in this phase.
        keys: set[str] = set()
        for record in train_err + val_err:
            if isinstance(record, dict):
                keys.update(record.keys())
        output_keys = sorted(keys)

        # Build aligned per‑key lists, filling missing entries with nan.
        train_err_aligned: Dict[str, List[float]] = defaultdict(list)
        val_err_aligned: Dict[str, List[float]] = defaultdict(list)

        for record in train_err:
            for k in output_keys:
                val = record.get(k) if isinstance(record, dict) else record
                train_err_aligned[k].append(val if val is not None else np.nan)
        for record in val_err:
            for k in output_keys:
                val = record.get(k) if isinstance(record, dict) else record
                val_err_aligned[k].append(val if val is not None else np.nan)

        # Determine maximum epoch count across all series.
        n_epochs = max(
            len(train_loss),
            len(val_loss),
            len(lr_hist),
            max((len(v) for v in train_err_aligned.values()), default=0),
            max((len(v) for v in val_err_aligned.values()), default=0),
        )
        epochs = list(range(n_epochs))

        # Pad lists to common length and replace None with nan.
        train_loss = _pad(
            [np.nan if v is None else v for v in train_loss], n_epochs)
        val_loss = _pad(
            [np.nan if v is None else v for v in val_loss], n_epochs)
        lr_hist = _pad([np.nan if v is None else v for v in lr_hist], n_epochs)
        for k in output_keys:
            train_err_aligned[k] = _pad(train_err_aligned[k], n_epochs)
            val_err_aligned[k] = _pad(val_err_aligned[k], n_epochs)

        aligned[phase] = {
            "epochs": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr_hist,
            "train_errors": train_err_aligned,
            "val_errors": val_err_aligned,
            "output_keys": output_keys,
        }

    return aligned

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def plot_training(history: Dict[str, Dict[str, list]], plot_config: dict[str, Any]) -> plt.Figure:
    """Plot training curves for each phase.

    Accepts either raw ``HistoryStorer`` dict or the output of
    :func:`align_epochs`. The function auto‑detects and aligns as needed.
    """
    # Auto‑align if necessary
    if "epochs" not in next(iter(history.values())):
        data = align_epochs(history)
    else:
        data = history  # type: ignore[assignment]

    n_phases = len(data)
    max_outputs = max(len(m["output_keys"]) for m in data.values())
    n_cols = 1 + max_outputs  # first column reserved for loss

    fig, axes = plt.subplots(
        nrows=n_phases,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_phases),
        squeeze=False,
    )

    for row, (phase, m) in enumerate(data.items()):
        epochs = m["epochs"]
        train_loss = np.asarray(m["train_loss"], dtype=float)
        val_loss = np.asarray(m["val_loss"], dtype=float)
        lr_hist = np.asarray(m["learning_rate"], dtype=float)

        # ---------------------- Loss column ----------------------
        ax_loss = axes[row][0]
        ax_loss.plot(epochs, train_loss, label="Train", lw=1.2, color='blue')
        if not np.isnan(val_loss).all():
            ax_loss.plot(epochs, val_loss, label="Val", lw=1.2, color='orange')
        ax_loss.set_title(f"{phase.capitalize()} – Loss")
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel(loss_label_map[plot_config['strategy']['loss']])
        ax_loss.set_yscale("log")
        ax_loss.legend()

        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, lr_hist, color="black", lw=0.5)
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_yscale("log")

        # -------------------- Error columns ----------------------
        for col, key in enumerate(m["output_keys"], start=1):
            train_err = np.asarray(m["train_errors"][key], dtype=float)
            val_err = np.asarray(m["val_errors"][key], dtype=float)

            ax = axes[row][col]
            if not np.isnan(train_err).all():
                ax.plot(epochs, train_err, label=f"Train",
                        lw=1.2, color='blue')
            if not np.isnan(val_err).all():
                ax.plot(epochs, val_err, label=f"Val", lw=1.2, color='orange')
            ax.set_title(f"{phase.capitalize()} – {key.capitalize()}")
            ax.set_xlabel('Epochs')
            ax.set_ylabel(error_label_map[plot_config['strategy']['error']])
            ax.set_yscale("log")
            ax.legend()

            ax_lr2 = ax.twinx()
            ax_lr2.plot(epochs, lr_hist, color="black", lw=0.5)
            ax_lr2.set_ylabel("Learning rate")
            ax_lr2.set_yscale("log")

    fig.tight_layout()
    return fig
