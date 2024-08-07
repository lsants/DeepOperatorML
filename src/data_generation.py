import numpy as np
import matplotlib.pyplot as plt
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

np.random.seed(42)

n = 4
m = 100

end = 2

def plot(x, o):
    y, G_y = o
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the first vector on the first subplot
    leg1 = []
    for i in range(len(y)):
        axes[0].plot(x, y[i])
        leg1.append(f'u{i}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].legend(leg1)

    leg2 = []
    for i in range(len(G_y)):
        axes[1].plot(x, G_y[i])
        leg2.append(f'G(u{i})(y)')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('G(u)(y)')
    axes[1].legend(leg2)
    
    plt.tight_layout()
    plt.show()


x = np.linspace(0, end, m)

# Defining input function
u_x = lambda r: r*x

def u(v):
    return np.array([np.array(u_x(r)) for r in v])

def G(y):
    return np.exp(-y)

random_point = np.random.choice(x)

# features
r = 10*np.random.rand(n)
u = u(r)

# labels

G_u = np.array([G(u_func[random_point == x][0]) for u_func in u])
output = u, G_u
# plot(x, output)

branch_features = u
branch_labels = G_u

# np.save(os.path.join(path_to_data, 'branch_features.npy'), branch_features)
# np.save(os.path.join(path_to_data, 'branch_features.npy'), branch_labels)

print(branch_features.shape, branch_labels.shape)