import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')


def G(a, b, y):
    c = (a/b).reshape(-1,1) # (n, 1) -> broadcasting row wise
    return c*np.sin(b*y.T)

def preprocess_data(u, y, Guy):
    branch_input = np.repeat(u, repeats=len(y), axis=0)
    trunk_input = np.tile(y, (len(u), 1))
    output = Guy.flatten().reshape(-1,1)
    return (branch_input, trunk_input, output)

def split_data(arr):
    train_spilt = 0.7
    size = len(arr)
    sample = int(train_spilt*size)
    train_data, test_data = arr[:sample, :], arr[sample:, :]
    return train_data, test_data

np.random.seed(42)

n = 100 # Number of input functions
m = 80 # Number of sensors (must be fixed)
q = 60 # Output locations (can be random)
p = 50 # Dimension of net's basis

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


# Split training and test set
data = np.concatenate(preprocess_data(u, y, G_u_y), axis=1)
antiderivative_train, antiderivative_test = split_data(data)

print(f"Train size: {antiderivative_train.shape}, \nTest size: {antiderivative_test.shape}")


if __name__ == '__main__':
    np.save(os.path.join(path_to_data, 'antiderivative_train.npy'), antiderivative_train)
    np.save(os.path.join(path_to_data, 'antiderivative_test.npy'), antiderivative_test)