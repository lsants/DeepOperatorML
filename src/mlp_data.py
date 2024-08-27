import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

np.random.seed(42)

def generate_data(n_samples):
    # polynomial coefficients (α, β, γ, B), in the interval [a, b]
    a = 0.1
    b = 1
    alpha_beta_gamma_B = np.random.uniform(a,b, 4*n_samples).reshape(-1,4)
    integrals = np.zeros((n_samples, 1))
    for i in range(n_samples):
        alpha, beta, gamma, b = alpha_beta_gamma_B[i]
        integrals[i] = (alpha / 3) * (b**3 - a**3) + (beta / 2) * (b**2 - a**2) + \
            gamma * (b - a)

    return alpha_beta_gamma_B, integrals

if __name__ == '__main__':
    data_path = os.path.join(project_dir, 'data')

    n = 20000
    train_size = 0.8
    test_size = 0.1
    np.random.seed(42)

    X, y = generate_data(n)
    poly_data = np.concatenate((X, y), axis=1)
    print('Dataset size:', '\n', poly_data.shape, '\n')

    # --------------------- split train-test set ---------------------
    train_data, test_data = train_test_split(poly_data, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=test_size, random_state=42)

    X_train, y_train = train_data[:, :-1].reshape(-1,4), train_data[:, -1].reshape(-1,1)
    X_val, y_val = val_data[:, :-1].reshape(-1,4), val_data[:, -1].reshape(-1,1)
    X_test, y_test = test_data[:, :-1].reshape(-1,4), test_data[:, -1].reshape(-1,1)

    np.savez(os.path.join(data_path, 'mlp_dataset_train.npz'),
              X=X_train, y=y_train,)
    np.savez(os.path.join(data_path, 'mlp_dataset_val.npz'),
              X=X_val, y=y_val)
    np.savez(os.path.join(data_path, 'mlp_dataset_test.npz'),
              X=X_test, y=y_test)
