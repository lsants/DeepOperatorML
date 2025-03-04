import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def align_epochs(history):
    """
    Aligns epochs across phases and aggregates metrics for plotting.

    Args:
        history (dict): Training and validation history across phases.

    Returns:
        dict: Dictionary with per-phase aligned data.
    """
    aligned_data = {}

    for phase, metrics in history.items():
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
        train_errors = metrics.get('train_errors', [])
        val_errors = metrics.get('val_errors', [])
        learning_rate = metrics.get('learning_rate', [])
        
        if train_errors and isinstance(train_errors[0], dict):
            output_keys = list(train_errors[0].keys())
        else:
            output_keys = []

        aligned_train_errors = {}
        aligned_val_errors = {}
        for key in output_keys:
            aligned_train_errors[key] = [e.get(key) for e in train_errors if isinstance(e, dict)]
            aligned_val_errors[key] = [e.get(key) for e in val_errors if isinstance(e, dict)]

        n_epochs = max(len(train_loss), len(val_loss))
        epochs = list(range(n_epochs))

        def extend_list(lst):
            return lst + [None] * (n_epochs - len(lst))

        train_loss = extend_list(train_loss)
        val_loss = extend_list(val_loss)
        for key in output_keys:
            aligned_train_errors[key] = extend_list(aligned_train_errors[key])
            aligned_val_errors[key] = extend_list(aligned_val_errors[key])
        learning_rate = extend_list(learning_rate)

        aligned_data[phase] = {
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'train_errors': aligned_train_errors,
            'val_errors': aligned_val_errors,
            'output_keys': output_keys
        }

    return aligned_data


def plot_training(history):
    """
    Plots training and validation metrics over epochs.
    
    For each phase, the first column shows the loss, and each subsequent column shows
    the error for one output (e.g. real, imaginary, or any other target as defined).
    
    Args:
        history (dict): Training and validation history, as output from align_epochs.
        
    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """
    n_phases = len(history)
    max_outputs = 0
    for phase, metrics in history.items():
        max_outputs = max(max_outputs, len(metrics.get('output_keys', [])))

    n_cols = 1 + max_outputs

    fig, axes = plt.subplots(nrows=n_phases, ncols=n_cols, figsize=(5 * n_cols, 5 * n_phases))


    if n_phases == 1:
        axes = [axes]

    for i, (phase, metrics) in enumerate(history.items()):
        epochs = metrics['epochs']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        learning_rate = metrics['learning_rate']
        output_keys = metrics.get('output_keys', [])

        # ----- Column 0: Loss plot -----
        ax_loss = axes[i][0] if n_cols > 1 else axes[i]
        ax_loss.plot(epochs, train_loss, label='Train Loss', color='blue')

        if any(v is not None for v in val_loss):
            ax_loss.plot(epochs, val_loss, label='Val Loss', color='orange')
        ax_loss.set_title(f"Phase: {phase} - Loss")
        ax_loss.set_yscale('log')
        ax_loss.legend()

        ax_loss_lr = ax_loss.twinx()
        ax_loss_lr.plot(epochs, learning_rate, label='Learning Rate', color='black', linewidth=0.5)
        ax_loss_lr.set_ylabel("Learning Rate")
        ax_loss_lr.set_yscale('log')

        # ----- Columns 1 and onward: Error plots per output key -----
        for col, key in enumerate(output_keys, start=1):
            ax = axes[i][col] if n_cols > 1 else axes[i]
            train_err = metrics['train_errors'][key]
            val_err = metrics['val_errors'][key]
            if any(e is not None for e in train_err):
                ax.plot(epochs, train_err, label=f"Train Error ({key})", color='blue')
            if any(e is not None for e in val_err):
                ax.plot(epochs, val_err, label=f"Val Error ({key})", color='orange')
            ax.set_title(f"Phase: {phase} - Error ({key})")
            ax.set_yscale('log')
            ax.legend()

            ax_lr = ax.twinx()
            ax_lr.plot(epochs, learning_rate, label='Learning Rate', color='black', linewidth=0.5)
            ax_lr.set_ylabel("Learning Rate")
            ax_lr.set_yscale('log')


    fig.tight_layout()
    return fig
