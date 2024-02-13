from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def split_dataset(data, TEST_SIZE=0.2, seed=0):
    """Split dataset in training and test sets.

    Args:
        data: dataset
        TEST_SIZE (float, optional): Fraction of dataset dedicated to training the model. Defaults to 0.2.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        split (tuple): training and validation sets
    """
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data[:][1],
        test_size=TEST_SIZE,
        random_state=seed
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    return train_split, test_split


def compute_integral(coefs):
    alpha, beta, gamma, B = coefs
    return (alpha / 3) * B**3 + (beta / 2) * B**2 + \
        gamma * B


def plot_performance(train_loss: list, test_loss: list, accuracy=None, model_name=None):
    """Generate plot with model's loss and accuracy curves.

    Args:
        train_loss (list): Training loss for each epoch.
        test_loss (list): Test loss for each epoch.
        accuracy (list, optional): Accuracy for each epoch. Defaults to None.
        model_name (str, optional): Name of tested model. Defaults to None.

    Returns:
        fig: Figure with plots.
    """
    if accuracy is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(train_loss, label='Training')
        ax[0].plot(test_loss, label='Validation')
        ax[0].set_ylabel(f"MSE", fontsize=9)
        # ax[0].set_ylim([0, 3e-1])
        ax[0].set_xlabel(f"Epochs/Iterations", fontsize=9)
        ax[0].set_title(f"Loss", fontsize=10)
        ax[0].ticklabel_format(style='scientific', scilimits=(-1, 2), axis='y',useMathText=True)
        ax[0].legend()

        ax[1].plot(accuracy, 'r', label='RMSE')
        ax[1].set_xlabel("Epochs/Iterations", fontsize=9)
        ax[1].set_ylabel("RMSE", fontsize=9)
        # ax[1].set_ylim([0, 1e-2])
        ax[1].set_title('Metric', fontsize=10)
        ax[1].ticklabel_format(style='scientific', scilimits=(-1, 2), axis='y',useMathText=True)
        ax[1].legend()
    else:
        fig, ax = plt.subplots()
        ax.plot(train_loss, label='Training')
        ax.plot(test_loss, label='Validation')
        ax.legend()
        ax.set_xlabel('Epochs/Iterations', fontsize=9)
        ax.set_title('Loss', fontsize=10)

    if model_name is not None:
        fig.suptitle(f"{model_name.replace('_', ', ')}", fontsize=11)
    return fig


def get_model_name(nodes_config: list, b_size: int, lr: float, epochs: int, data_size=None) -> str:
    batch_size = f'Batchsize={b_size}'
    val_lr = f'LR={lr:.0e}'
    n_epochs = f'Epochs={epochs}'
    schema = '->'.join(map(str, ['X'] + nodes_config + ['yp']))
    samples = 'FullDataset'
    if data_size is not None:
        samples = f'Samples={data_size}'

    params = [samples, n_epochs, batch_size, val_lr, schema]
    return '_'.join(params)

def normalize(poly, X_mean, X_std)-> np.array:
    if type(poly) != np.ndarray:
        poly = np.array(poly)
    return (poly - X_mean)/X_std

def unnormalize(y_n: np.array, y_mean, y_std)->np.array:
    return y_n*y_std + y_mean