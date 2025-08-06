from __future__ import annotations
from collections import defaultdict
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)

error_label_map = {
    'vector_l2': r'$L_{2}$',
    'matrix_fro': r'$L_{F}$',
}

loss_label_map = {
    'mse': f'MSE',
    'huber': f'Huber',
    'mse_dissipative': '$\\mathcal{L}_{\\lambda}$',
    'wasserstein': f'$W$'
}

# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def _pad(seq: list[Any], n: int, pad_val: Any = np.nan) -> list[Any]:
    """Right‑pad *seq* to length *n* with *pad_val*."""
    return seq + [pad_val] * (n - len(seq))


def align_epochs(raw_history: dict[str, dict[str, list]]) -> dict[str, dict[str, Any]]:
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
    aligned: dict[str, dict[str, Any]] = {}

    for phase, metrics in raw_history.items():
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])
        lr_hist = metrics.get("learning_rate", [])
        train_err = metrics.get("train_errors", [])
        val_err = metrics.get("val_errors", [])

        keys: set[str] = set()
        for record in train_err + val_err:
            if isinstance(record, dict):
                keys.update(record.keys())
        output_keys = sorted(keys)

        train_err_aligned: dict[str, list[float]] = defaultdict(list)
        val_err_aligned: dict[str, list[float]] = defaultdict(list)

        for record in train_err:
            for k in output_keys:
                val = record.get(k) if isinstance(record, dict) else record
                train_err_aligned[k].append(val if val is not None else np.nan)
        for record in val_err:
            for k in output_keys:
                val = record.get(k) if isinstance(record, dict) else record
                val_err_aligned[k].append(val if val is not None else np.nan)

        n_epochs = max(
            len(train_loss),
            len(val_loss),
            len(lr_hist),
            max((len(v) for v in train_err_aligned.values()), default=0),
            max((len(v) for v in val_err_aligned.values()), default=0),
        )
        epochs = list(range(n_epochs))

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


def plot_training(
    history: dict[str, dict],
    plot_config: dict[str, Any]
) -> Figure:
    """
    history: output of HistoryStorer.get_history()
    plot_config: same as before, for labels/scales
    """
    phases = list(history.keys())
    n_phases = len(phases)

    max_err = max(len(h['train_errors']) for h in history.values())
    n_cols = 1 + max_err  # first col for loss

    fig, axes = plt.subplots(
        nrows=n_phases,
        ncols=n_cols,
        figsize=(5 * n_cols, 4.5 * n_phases),
        squeeze=False
    )

    for row, phase in enumerate(phases):
        h = history[phase]
        # x-axis
        if len(phase) == 3:
            phase = phase.upper()
        else:
            phase = phase.capitalize()
        phase = phase.replace('_', ' ')
        epochs = np.arange(1, len(h['train_loss']) + 1)

        # --- Loss plot ---
        ax_loss = axes[row][0]
        ax_loss.plot(epochs, h['train_loss'], label='Train', lw=1.3)
        if h['val_loss']:
            ax_loss.plot(epochs, h['val_loss'], label='Val', lw=1.3)
        ax_loss.set_title(f"{phase} – Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel(loss_label_map[plot_config['strategy']['loss']])
        ax_loss.set_yscale("log")
        ax_loss.legend()

        # LR on same plot
        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, h['learning_rate'], color='k', lw=0.5)
        ax_lr.set_ylabel("LR")
        ax_lr.set_yscale("log")

        # --- Error plots ---
        err_keys = list(h['train_errors'].keys())
        for col, key in enumerate(err_keys, start=1):
            title = key.replace('Error_', 'Error ')
            ax = axes[row][col]
            train_err = np.asarray(h['train_errors'][key], dtype=float)
            val_err = np.asarray(h['val_errors'].get(key, []), dtype=float)

            ax.plot(epochs, train_err, label='Train', lw=1.3)
            if val_err.size:
                ax.plot(epochs, val_err, label='Val', lw=1.3)

            ax.set_title(f"{phase} – {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(error_label_map[plot_config['strategy']['error']])
            ax.set_yscale("log")
            ax.legend()

            # LR again
            ax2 = ax.twinx()
            ax2.plot(epochs, h['learning_rate'], color='k', lw=0.5)
            ax2.set_ylabel("LR")
            ax2.set_yscale("log")

        # blank out any unused subplots
        for empty_col in range(1 + len(err_keys), n_cols):
            axes[row][empty_col].axis('off')

    fig.tight_layout()
    return fig
