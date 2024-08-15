import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

np.random.seed(42)

def G(a, b, y):
    c = (a/b).reshape(-1,1) # (n, 1) -> broadcasting row wise
    return c*np.sin(b*y.T)

def split_data(arr):
    train_spilt = 0.7
    size = len(arr)
    sample = int(train_spilt*size)
    train_data, test_data = arr[:sample, :], arr[sample:, :]
    return train_data, test_data

# ----------- Set size of dataset and operator domain -----------
n = 100 # Number of input functions
m = 80 # Number of sensors (must be fixed)
q = 60 # Output locations (can be random)
start = 0
end = 1000

# ------- Branch input ------
x = np.linspace(start, end, m) # sensors (dont necessarily have to be on a lattice - can be random - linspace is not required)
a , b = 10*np.random.rand(n).reshape(n, 1), 10*np.random.rand(n).reshape(n, 1) 
u = np.array([a[i] * np.cos(b[i]*x) for i in range(n)]) # function is u_n = a_n * cos(b_n*x)

# ------- Trunk input -------
y = end *np.random.rand(q).reshape(q, 1)

# ------- Output -----------
G_u_y = G(a, b, y)

# _------ Split training and test set --------
u_train, u_test = split_data(u)
y_train, y_test = split_data(y)
G_train, G_test = split_data(G_u_y)

train_data = (u_train, y_train, G_train)
test_data = (u_test, y_test, G_test)

train_shapes = '\n'.join([str(i.shape) for i in train_data])
test_shapes = '\n'.join([str(i.shape) for i in test_data])

print(f"Train sizes: \n{train_shapes}, \nTest size: {test_shapes}")

# --- Save dataset ---
if __name__ == '__main__':
    np.savez(os.path.join(path_to_data, 'antiderivative_train.npz'), X_branch=u_train, X_trunk=y_train, y=G_train)
    np.savez(os.path.join(path_to_data, 'antiderivative_test.npz'), X_branch=u_test, X_trunk=y_test, y=G_test)