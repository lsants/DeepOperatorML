# ----------------- Imports and path setup ----------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')

pd.set_option('display.max_columns', 500)

def extract_real_imag_parts(arr: np.ndarray[np.complex128]):
    real_part = arr.real
    imaginary_part = arr.imag
    return np.stack((real_part, imaginary_part), axis=-1)

def remove_unused_features(arr):
    return arr[:, -1].reshape(-1,1)

def preprocess(arr):
    out = remove_unused_features(arr)
    return out

# -------------------- Train data ------------------------
d_train = np.load(f"{path_to_data}/iss_train_fixed.npz", allow_pickle=True)
u_train_full, trunk_features, labels_train_r, labels_train_z = d_train['X_branch'],d_train['X_trunk'], d_train['y_r'], d_train['y_z']

u_train = preprocess(u_train_full)
labels_train_r = extract_real_imag_parts(labels_train_r)
labels_train_z = extract_real_imag_parts(labels_train_z)

# -------------------- Test data ------------------------
d_test = np.load(f"{path_to_data}/iss_test_fixed.npz", allow_pickle=True)
u_test_full, trunk_features, labels_test_r, labels_test_z = d_test['X_branch'],d_test['X_trunk'], d_test['y_r'], d_test['y_z']

u_test = preprocess(u_test_full)
labels_test_r = extract_real_imag_parts(labels_test_r)
labels_test_z = extract_real_imag_parts(labels_test_z)

# # ----------------------------- Saving data -------------------------------------
# if __name__ == '__main__':
#     np.savez(os.path.join(path_to_data, 'iss_train_processed.npz'), X_branch=u_train, X_trunk=trunk_features, y_r=labels_train_r, y_z=labels_train_z)
#     np.savez(os.path.join(path_to_data, 'iss_test_processed.npz'), X_branch=u_test, X_trunk=trunk_features, y_r=labels_test_r, y_z=labels_test_z)