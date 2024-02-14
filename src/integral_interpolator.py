# ------------ Integration by Interpolation ---------------
''' 
We want to use a NN to compute the integral I of a polynomial function in the form of:
    αx^2 + βx + γ, from zero to B ->
    I = α*B^3/3 + β*B^2/2 + γ*B
'''
# --------------------- Modules ---------------------
import os
import sys
sys.path.insert(0, '/home/lsantiago/workspace/ic/project/')
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from src.generate_poly_dataset import PolyDataset, ToTensor
from src import generic as gc
from src import model_training as mt
from src import nn_architecture as NN
import torch
import numpy as np

# --------------------- Paths ---------------------
working_path = sys.path[0]
path_to_data = os.path.join(working_path, 'data')
path_to_models = os.path.join(working_path, 'models')
path_to_images = os.path.join(working_path, 'images')
data = os.path.join(path_to_data, 'poly_data.npy')

# --------------------- Parameters ---------------------
torch.set_default_dtype(torch.float64)
batch_size = 100
lr = 0.001
epochs = 700
n_sample = 40000
full_data = False

# --------------------- Get and normalize dataset ---------------------
poly_data = np.load(data)  # Load polynomials numpy array
# poly_data[:, -2] = 1 # Fixing limit of integration
poly_data = poly_data.astype(np.float64)
mean = np.mean(poly_data, axis=0)
y_mean = mean[-1]
X_mean = mean[:-1]
std = np.std(poly_data, axis=0)
y_std = std[-1]
X_std = std[:-1]
poly_data_norm = (poly_data - mean)/std
poly_data = poly_data_norm
if not full_data:
    poly_data = poly_data[:n_sample]

# --------------------- Format as torch and split train-test set ---------------------
poly_data = poly_data[:, :-1], poly_data[:, -1]
data_size = len(poly_data[0]) if not full_data else None
transformed_dataset = PolyDataset(poly_data, transform=ToTensor())
X, y = transformed_dataset.data
print(X.shape, y.shape)
training_data, test_data = gc.split_dataset(transformed_dataset, seed=42)

# --------------------- Create data loaders ---------------------
train_dataloader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True)

# --------------------- Define NN architecture ---------------------
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")
model = NN.NeuralNetwork().to(device)
nodes_config = NN.nodes_config
# print(model)

# --------------------- Defining loss function and optimizer ---------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --------------------- Get model filename ---------------------
if full_data:
    data_size = None
model_name = gc.get_model_name(nodes_config, batch_size, lr, epochs, data_size)
plot_path = os.path.join(path_to_images, model_name)

# --------------------- Train/test the model ---------------------
tl = []
vl = []
acc = []
early_stopper = mt.EarlyStopper(patience=5, min_delta=1e-2)
for t in tqdm(range(epochs)):
    train_loss = mt.train(train_dataloader, model, loss_fn, optimizer)
    test_loss, metric = mt.test(train_dataloader, model, loss_fn)
    tl.append(train_loss)
    vl.append(test_loss)
    acc.append(metric)
    # if t % 1 == 0:
        # print(f"Epoch {t}\n-------------------------")
        # print(f"Test error: \n Performance {(metric):>0.4e}, Avg train loss: {train_loss:>8e}, Avg test loss: {test_loss:>8e} \n")
    if early_stopper.early_stop(test_loss):
        print(f"Early stopping at:\nEpoch {t}")
        break
print("Done!\n")

# --------------------- Plot loss curves ---------------------
plots = gc.plot_performance(
    train_loss=tl, test_loss=vl, accuracy=acc, model_name=model_name)
print(
    f"For training: \n Min RMSE {(np.array(acc).min()):>0.4e}, Min loss: {np.array(test_loss).min():>8e} \n")
plt.savefig(f"{plot_path}.png")

# --------------------- Save model ---------------------
torch.save(model.state_dict(), os.path.join(path_to_models, model_name))

# --------------------- Testing ---------------------
X_n, y_n = poly_data[0], poly_data[1]
y = y_n*y_std + y_mean
y_pred_n = model(torch.atleast_2d(torch.tensor(X_n))
                 ).detach().numpy().squeeze()
y_pred = y_pred_n*y_std + y_mean
y_train_n = training_data[:][1]
y_train = y_train_n*y_std + y_mean
y_test_n = test_data[:][1]
y_test = y_test_n*y_std + y_mean
error = (np.abs(y_pred - y) < 0.1*y).sum() / len(y)
print(f'Accuracy on training set for 10% error:\n{error:.2%}')

a = gc.compute_integral(([0, 0, 1, 1]))
b = model(torch.atleast_2d(torch.tensor(gc.normalize(
    [0, 0, 1, 1], X_mean, X_std), dtype=torch.float64)))
b = gc.unnormalize(b.item(), y_mean, y_std,)
outs = [a,
        b,
        ((a - b)/a).item()]
outs = list(map(lambda x: f'{x:.3f}', outs))
outs[-1] = f'{float(outs[-1]):.1%}'
print(*outs, '\n')

fig1, ax = plt.subplots(2, 2, figsize=(8, 4))
ax[0, 0].hist(y, bins=1000, label='y', color='blue', density=True)
ax[0, 0].set_xlim([0, 2])
ax[0, 0].set_title(f'$y$')
ax[0, 1].hist(y_pred, bins=1000, label='y_pred',
              color='darkorange', density=True)
ax[0, 1].set_xlim([0, 2])
ax[0, 1].set_title(f'$y_{{pred}}$')
ax[1, 0].hist(y_train, bins=1000, label='y_train', color='green', density=True)
ax[1, 0].set_xlim([0, 2])
ax[1, 0].set_title(f'$y_{{train}}$')
ax[1, 1].hist(y_test, bins=1000, label='y_test', color='green', density=True)
ax[1, 1].set_xlim([0, 2])
ax[1, 1].set_title(f'$y_{{test}}$')
fig1.tight_layout()


plt.show()
