import numpy as np
import logging

logger = logging.getLogger(__file__)

def pod_stacked_data(data: np.ndarray, var_share: float) -> tuple[np.ndarray, ...]:
    n_samples, n_space, n_channels = data.shape
    snapshots = data.swapaxes(1, 2).reshape(
        n_samples * n_channels, n_space)  # (n_samples * n_channels, n_space))

    mean_field = np.mean(snapshots, axis=0, keepdims=True)  # (1, n_space)
    centered = (snapshots - mean_field)
    W, S, Vt = np.linalg.svd(centered, full_matrices=False)
    spatial_vectors = Vt.T

    explained_variance_ratio = np.cumsum(
        S**2) / np.linalg.norm(S, ord=2)**2
    
    n_vectors = max(1, (explained_variance_ratio < var_share).sum().item())

    logger.info(f"Dataset has {n_vectors} vectors")


    basis = spatial_vectors[:, : n_vectors]  # (n_space, n_vectors)

    # import matplotlib.pyplot as plt

    # i = 0
    # for v in basis.T:
    #     v = np.concatenate(
    #         (np.flip(v.reshape(40, 40), axis=0), v.reshape(40, 40)), axis=0)
    #     plt.contourf(np.flipud(v.T), cmap="viridis")
    #     plt.colorbar()
    #     plt.show()
    #     i += 1
    #     if i >= 10:
    #         break
    # quit()
    # for snap in snapshots:
    #     snap = np.concatenate(
    #         (np.flip(snap.reshape(40, 40), axis=0), snap.reshape(40, 40)), axis=0)
    #     plt.contourf(np.flipud(snap.T), cmap="viridis")
    #     plt.colorbar()
    #     plt.show()
    # quit()

    return basis, mean_field


def pod_split_data(
    data: np.ndarray,              # (N_s, N_r*N_z, N_c)
    var_share: float
) -> tuple[np.ndarray, np.ndarray]:

    n_samp, n_space, n_chan = data.shape
    mean_field = np.empty((n_chan, n_space))
    vectors_list: list[np.ndarray] = []

    for c in range(n_chan):
        snapshots = data[:, :, c]                      # (N_samp, n_space)
        mean_c = snapshots.mean(axis=0)                # (n_space,)
        mean_field[c] = mean_c
        A = snapshots - mean_c                         # centred

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T                                       # (n_space, rank)

        cum_var = np.cumsum(S**2) / np.sum(S**2)
        n_vectors_c = np.searchsorted(cum_var, var_share) + 1

        logger.info(f"Channel {c + 1} has {n_vectors_c} vectors")

        # (n_space, n_vectors_c)
        vectors_c = V[:, : n_vectors_c]
        vectors_list.append(vectors_c)

    min_n_vectors = min(v.shape[1] for v in vectors_list)
    adjusted_vectors_list = [v[:, : min_n_vectors] for v in vectors_list]
    basis = np.concatenate(adjusted_vectors_list, axis=1)
    return basis, mean_field
