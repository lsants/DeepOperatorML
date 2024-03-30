from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import os
import sys
script_dir = os.path.dirname(os.getcwd())
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')
path_to_images = os.path.join(project_dir, 'images')
norm_params = np.load('/home/lsantiago/workspace/ic/project/models/normalization_params.npz')
norm_mean, norm_std = norm_params['mean'], norm_params['std']

# --------------------- Get dataset ---------------------



def split_dataset(data, VAL_SIZE=0.1, TEST_SIZE=0.1, seed=42):
    """Split dataset into training, validation, and test sets.

    Args:
        data: dataset
        VAL_SIZE (float, optional): Fraction of dataset dedicated to validation. Defaults to 0.2.
        TEST_SIZE (float, optional): Fraction of dataset dedicated to testing. Defaults to 0.2.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        split (tuple): training, validation, and test sets
    """
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data[:, 1],
        test_size=TEST_SIZE,
        random_state=seed
    )
    # Split the training indices into training and validation indices
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=seed
    )

    # generate subset based on indices
    train_split = data[train_indices]
    val_split = data[val_indices]
    test_split = data[test_indices]

    return train_split, val_split, test_split


def compute_integral(coefs):
    integrals = []
    for poly in coefs:
        if len(poly) == 3:
            alpha, beta, gamma = poly
            B = 1
        else:
            alpha, beta, gamma, B = poly
        integrals.append((alpha / 3) * B**3 + (beta / 2) * B**2 +
                         gamma * B)
    return integrals


def plot_loss(train_loss: list, val_loss: list, model_name=None):
    """Generate plot with model's loss curves.

    Args:
        train_loss (list): Training loss for each epoch.
        test_loss (list): Test loss for each epoch.

    Returns:
        fig: Figure with plots.
    """
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Training')
    ax.plot(val_loss, label='Validation')
    ax.legend()
    ax.set_xlabel('Epochs/Iterations', fontsize=9)
    ax.ticklabel_format(style='scientific', scilimits=(-1,
                        2), axis='y', useMathText=True)
    ax.set_yscale('log')
    ax.set_title('Loss', fontsize=10)

    if model_name is not None:
        fig.suptitle(f"{model_name.replace('_', ', ')}", fontsize=11)
    return fig


def plot_histograms(*args):
    y, y_train, y_val, y_test, y_pred = args
    fig1, ax = plt.subplots(2, 3, figsize=(8, 4))
    ax[0, 0].hist(y, bins=1000, label='y', color='blue', density=True)
    ax[0, 0].set_xlim([-2, 2])
    ax[0, 0].set_title(f'$y$')
    ax[0, 1].hist(y_pred, bins=1000,
                  label=f'y_{{pred_{{test}}}}', color='darkorange', density=True)
    ax[0, 1].set_xlim([-2, 2])
    ax[0, 1].set_title(f'$y_{{pred}}$')
    ax[1, 0].hist(y_train, bins=1000,
                  label=f'y_{{train}}', color='green', density=True)
    ax[1, 0].set_xlim([-2, 2])
    ax[1, 0].set_title(f'$y_{{train}}$')
    ax[1, 1].hist(y_val, bins=1000,
                  label=f'y_{{val}}', color='green', density=True)
    ax[1, 1].set_xlim([-2, 2])
    ax[1, 1].set_title(f'$y_{{val}}$')
    ax[1, 2].hist(y_test, bins=1000,
                  label=f'y_{{test}}', color='green', density=True)
    ax[1, 2].set_xlim([-2, 2])
    ax[1, 2].set_title(f'$y_{{test}}$')
    fig1.tight_layout()
    return fig1


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


def normalize(X, mean=norm_mean, std=norm_std):
    if type(X) == torch.Tensor:
        X = X.detach().cpu().numpy()
        return torch.tensor((X - mean)/std)
    else:
      return (X - mean)/std


def predict(model, X, train_mean=norm_mean, train_std=norm_std) -> np.ndarray:
    X = normalize(X, train_mean, train_std)
    model.eval()
    if type(X) == list:
        X = np.array(X)
    X = torch.atleast_2d(torch.tensor(X))
    with torch.no_grad():
        prediction = model(X).detach().numpy()
        return prediction.squeeze()
