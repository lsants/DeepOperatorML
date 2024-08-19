import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

np.random.seed(42)

def G(a, b, y):
    c = (a/b).reshape(-1,1) # (n, 1) -> broadcasting row wise
    return c*np.sin(b*y.T)

# ----------- Set size of dataset and operator domain -----------
n, m, = 300, 80 # Number of input functions and sensors (must be fixed)
q = 500 # Output locations (can be random)
start = 0
end = 3*np.pi
# ------- Branch input ------
x = np.linspace(start, end, m) # sensors (dont necessarily have to be on a lattice - can be random - linspace is not required)
a , b = np.random.rand(n).reshape(n, 1), np.random.rand(n).reshape(n, 1) 
u = np.array([a[i] * np.cos(b[i]*x) for i in range(n)]) # function is u_n = a_n * cos(b_n*x)

# ------- Trunk input -------
y = end *np.random.rand(q).reshape(q, 1)

# ------- Output -----------
G = G(a, b, y)

# _------ Split training and test set --------
test_size = 0.3
train_rows = int(n * (1 - test_size))
test_rows = n - train_rows           

train_cols = int(q * (1 - test_size))
test_cols = q - train_cols           

u_train, u_test, G_train, G_test = train_test_split(u, G, test_size=test_size, random_state=42)

train_data = (u_train, G_train)
test_data = (u_test, G_test)

train_shapes = '\n'.join([str(i.shape) for i in train_data])
test_shapes = '\n'.join([str(i.shape) for i in test_data])

print(f"Train sizes (u, G): \n{train_shapes}, \nTest sizes (u, G): \n{test_shapes}")

# --- Save dataset ---
if __name__ == '__main__':
    np.savez(os.path.join(path_to_data, 'antiderivative_train.npz'), X_branch=u_train, X_trunk=y, y=G_train, sensors=x)
    np.savez(os.path.join(path_to_data, 'antiderivative_test.npz'), X_branch=u_test, X_trunk=y, y=G_test, sensors=x)