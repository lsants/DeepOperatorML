import matplotlib.pyplot as plt

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
        # Align epochs within the phase
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
        train_errors = metrics.get('train_errors', [])
        val_errors = metrics.get('val_errors', [])
        learning_rate = metrics.get('learning_rate', [])

        # Extract real and imaginary errors if present
        train_errors_real = [e.get('g_u_real') for e in train_errors if isinstance(e, dict)]
        train_errors_imag = [e.get('g_u_imag') for e in train_errors if isinstance(e, dict)]
        val_errors_real = [e.get('g_u_real') for e in val_errors if isinstance(e, dict)]
        val_errors_imag = [e.get('g_u_imag') for e in val_errors if isinstance(e, dict)]

        # Ensure epochs are aligned properly
        num_epochs = max(len(train_loss), len(val_loss), len(train_errors_real), len(train_errors_imag))
        epochs = list(range(num_epochs))

        # Adjust lengths to ensure alignment
        train_loss = train_loss + [None] * (num_epochs - len(train_loss))
        val_loss = val_loss + [None] * (num_epochs - len(val_loss))
        train_errors_real = train_errors_real + [None] * (num_epochs - len(train_errors_real))
        train_errors_imag = train_errors_imag + [None] * (num_epochs - len(train_errors_imag))
        val_errors_real = val_errors_real + [None] * (num_epochs - len(val_errors_real))
        val_errors_imag = val_errors_imag + [None] * (num_epochs - len(val_errors_imag))
        learning_rate = learning_rate + [None] * (num_epochs - len(learning_rate))

        aligned_data[phase] = {
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_errors_real': train_errors_real,
            'train_errors_imag': train_errors_imag,
            'val_errors_real': val_errors_real,
            'val_errors_imag': val_errors_imag,
            'learning_rate': learning_rate,
        }

    return aligned_data


def plot_training(history):
    """
    Plots training and validation metrics over epochs.

    Args:
        history (dict): Training and validation history, phase-specific.
    """
    n_phases = len(history)
    fig, ax = plt.subplots(nrows=n_phases, ncols=3, figsize=(15, 5 * n_phases))

    if n_phases == 1:
        ax = [ax]

    for i, (phase, metrics) in enumerate(history.items()):
        epochs = metrics['epochs']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_errors_real = metrics['train_errors_real']
        train_errors_imag = metrics['train_errors_imag']
        val_errors_real = metrics['val_errors_real']
        val_errors_imag = metrics['val_errors_imag']
        learning_rate = metrics['learning_rate']

        ax[i][0].plot(epochs[:len(train_loss)], train_loss, label='Train Loss', color='blue')
        if val_loss:
            ax[i][0].plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', color='orange')
        ax[i][0].set_title(f"Phase: {phase} - Loss")
        ax[i][0].set_yscale('log')
        ax[i][0].legend()
        ax_i_0 = ax[i][0].twinx()
        ax_i_0.plot(epochs[:len(train_loss)], learning_rate, label='Train Loss', color='black', linewidth=0.5)
        ax_i_0.set_ylabel(r"learning_rate")
        ax_i_0.set_yscale(r"log")

        if train_errors_real:
            ax[i][1].plot(epochs[:len(train_errors_real)], train_errors_real, label='Train Error (Real)', color='blue')
        if val_errors_real:
            ax[i][1].plot(epochs[:len(val_errors_real)], val_errors_real, label='Validation Error (Real)', color='orange')
        ax[i][1].set_title(f"Phase: {phase} - $L_2$ Error (Real)")
        ax[i][1].set_yscale('log')
        ax[i][1].legend()
        ax_i_1 = ax[i][1].twinx()
        ax_i_1.plot(epochs[:len(train_loss)], learning_rate, label='Train Loss', color='black', linewidth=0.5)
        ax_i_1.set_ylabel(r"learning_rate")
        ax_i_1.set_yscale(r"log")

        if train_errors_imag:
            ax[i][2].plot(epochs[:len(train_errors_imag)], train_errors_imag, label='Train Error (Imag)', color='blue')
        if val_errors_imag:
            ax[i][2].plot(epochs[:len(val_errors_imag)], val_errors_imag, label='Validation Error (Imag)', color='orange')
        ax[i][2].set_title(f"Phase: {phase} - $L_2$ Error (Imag)")
        ax[i][2].set_yscale('log')
        ax_i_2 = ax[i][2].twinx()
        ax[i][2].legend()
        ax_i_2 = ax[i][2].twinx()
        ax_i_2.plot(epochs[:len(train_loss)], learning_rate, label='Train Loss', color='black', linewidth=0.5)
        ax_i_2.set_ylabel(r"learning_rate")
        ax_i_2.set_yscale(r"log")

    fig.tight_layout()
    return fig
