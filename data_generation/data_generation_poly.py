import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

np.random.seed(42)

def G(params, y):
    a,b,c = list(map(lambda x: x.reshape(-1,1), params))
    return (a/3)*y.T**3 + (b/2)*y.T**2 + c*y.T

# ----------- Set size of dataset and operator domain -----------
n = 300 # Number of input functions
q = 500 # Output locations (can be random)
start = 0.1
end = 1
n_params = 3
e = 1/(start - end)

# ------- Branch input ------
params = np.random.uniform(start - e, end + e, n_params*n).reshape(-1, n_params)
u = params

# ------- Trunk input -------
y = np.random.uniform(start, end, q).reshape(q, 1)

# ------- Output -----------
G = G(u.T, y)

# _------ Split training and test set --------
test_size = 0.2
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
    np.savez(os.path.join(path_to_data, 'antiderivative_train.npz'), X_branch=u_train, X_trunk=y, y=G_train, sensors=y)
    np.savez(os.path.join(path_to_data, 'antiderivative_test.npz'), X_branch=u_test, X_trunk=y, y=G_test, sensors=y)