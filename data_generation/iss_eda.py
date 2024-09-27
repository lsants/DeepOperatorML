# ----------------- Imports and path setup ----------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sc
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

pd.set_option('display.max_columns', 500)

# -------------------- Load data ------------------------
d = np.load(f"{path_to_data}/iss_train_processed.npz", allow_pickle=True)
X_b_full,X_t, y_r, y_z = d['X_branch'],d['X_trunk'], d['y_r'], d['y_z']

X_b = X_b_full[:, -1].reshape(-1,1)

X_b_headers = ['freq']
X_t_headers = ['r', 'z']
y_headers = ['Re(ur)', 'Im(ur)', 'Re(uz)', 'Im(uz)']

df_b = pd.DataFrame(columns=X_b_headers)
df_t = pd.DataFrame(columns=X_t_headers)
for index,name in enumerate(df_b.columns):
    df_b[name] = X_b.T[index]
for index,name in enumerate(df_t.columns):
    df_t[name] = X_t.T[index]

df_b = df_b.astype(float)
df_t = df_t.astype(float)

r,z = list(map(lambda x: x[-1], df_t[['r', 'z']].items()))
uz_real = y_z[0,:,0]
uz_imag = y_z[0,:,1]

print(r.shape,z.shape)
print(uz_real.shape,uz_imag.shape)


fig = plt.figure(figsize=plt.figaspect(0.5))

R, Z = np.meshgrid(r,z)
ax = fig.add_subplot(1, 2, 1, projection='3d')
aaa = ax.plot_surface(R, Z, uz_real, rstride=1, cstride=1)
bbb = ax.plot_surface(R, Z, uz_imag, rstride=1, cstride=1)
plt.show()