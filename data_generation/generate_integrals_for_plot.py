# ----------------- Imports and path setup ----------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm.auto import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
path_to_data = os.path.join(project_dir, 'data')
from high_performance_integration.src.influence_function import kernel_r, kernel_z

pd.set_option('display.max_columns', 500)

def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def integrate_kernels(branch_vars, trunk_vars, lower_bound, upper_bound):
    integrals_r = []
    integrals_z = []
    errors_r = []
    errors_z = []
    n = len(branch_vars)
    q = len(trunk_vars)
    E, ν, ρ, ω = branch_vars.T
    point = trunk_vars
    for i in tqdm(range(n), colour='GREEN'):
        for j in range(q):
            instance = (E[i], ν[i], ρ[i], ω[i])
            p = point[j]
            integral_r, error_r = integrate.quad(lambda ζ: ζ*kernel_r(ζ, instance, p), lower_bound, upper_bound, complex_func=True)
            integral_z, error_z = integrate.quad(lambda ζ: ζ*kernel_z(ζ, instance, p), lower_bound, upper_bound, complex_func=True)
            integrals_r.append(integral_r)
            integrals_z.append(integral_z)
            errors_r.append(error_r)
            errors_z.append(error_z)
    integrals_r = np.array(integrals_r).reshape(n, q)
    integrals_z = np.array(integrals_z).reshape(n, q)
    return np.array(integrals_r), np.array(integrals_z), np.array(errors_r), np.array(errors_z)


# ---------------- Get data ---------------
d_train = np.load(f"{path_to_data}/iss_train_fixed.npz", allow_pickle=True)
u_train_full, trunk_features, labels_train_r, labels_train_z = d_train['X_branch'],d_train['X_trunk'], d_train['y_r'], d_train['y_z']
