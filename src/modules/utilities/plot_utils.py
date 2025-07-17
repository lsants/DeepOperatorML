from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING, Any, Iterable
if TYPE_CHECKING:
    from modules.model.deeponet import DeepONet

logger = logging.getLogger(__name__)


def format_param(param: dict[str, Any] | Iterable, param_keys: list[str] | tuple | None = None) -> str:
    """
    Format a parameter value for display in the plot title.

    - If 'param' is a dict, it returns a string of the form:
          (key1=value1, key2=value2, ...)
    - If 'param' is an iterable (but not a string) and param_keys is provided and its length
      matches the length of 'param', it returns a string of the form:
          (key1=value1, key2=value2, ...)
      where the keys are taken from param_keys.
    - Otherwise, it returns the string representation of param.

    Args:
        param: The parameter value, which can be a dict, tuple, list, etc.
        param_keys (list or tuple, optional): List of keys to use if 'param' is an iterable.

    Returns:
        str: The formatted parameter string.
    """
    if isinstance(param, dict):
        items = [f"{k}={v:.1E}" for k, v in param.items()]
        return "(" + ", ".join(items) + ")"
    elif hasattr(param, '__iter__') and not isinstance(param, str):
        if param_keys is not None and len(param_keys) == len(param):
            items = [f"{k}={v:.1E}" for k, v in zip(param_keys, param)]
            return "(" + ", ".join(items) + ")"
        else:
            items = [f"{v:.1E}" for v in param]
            return "(" + ", ".join(items) + ")"
    else:
        return f"{param:.3f}"


def get_modes_indices_to_highlight(abs_mean_coeffs: np.ndarray, n: int) -> np.ndarray:
    n_channels = abs_mean_coeffs.shape[-1]
    mode_indices = np.array(
        [np.argpartition(abs_mean_coeffs[:,  j], -n)[-n:] for j in range(n_channels)]).T
    return np.sort(mode_indices, axis=0)


def flip_sign_of_negative_modes(modes: np.ndarray, mean_coeffs: np.ndarray) -> np.ndarray:
    inverted_modes = modes.copy()  # (i, ..., j)
    n_channels_basis, n_channels_coeffs = modes.shape[-1], mean_coeffs.shape[-1]
    if n_channels_basis == n_channels_coeffs:  # (n_basis = n_coeffs)
        flip_sign_mask = mean_coeffs < 0  # (i, j)
        n_dims_to_expand = inverted_modes.ndim - flip_sign_mask.ndim
        dims_dummy_array = [1 for _ in range(n_dims_to_expand)]
        flip_sign_mask_expanded = flip_sign_mask.reshape(
            inverted_modes.shape[0], *dims_dummy_array, inverted_modes.shape[-1])
        inverted_modes *= np.where(flip_sign_mask_expanded, -1, 1)
    else:
        flip_sign_masks = np.array(
            [mean_coeffs[..., i] < 0 for i in range(n_channels_coeffs)])
        n_dims_to_expand = inverted_modes.ndim - flip_sign_masks.ndim
        dims_dummy_array = [1 for _ in range(n_dims_to_expand)]
        flip_sign_masks_expanded = [mask.reshape(
            inverted_modes.shape[0], *dims_dummy_array, inverted_modes.shape[-1]) for mask in flip_sign_masks]
        final_mask = np.full(
            (flip_sign_masks_expanded[0].shape), False, dtype=bool)
        for mask in flip_sign_masks_expanded:
            final_mask |= mask
        inverted_modes *= np.where(final_mask, -1, 1)
    return inverted_modes


def mirror(arr: np.ndarray) -> np.ndarray:
    arr_flip = np.flip(arr[1:, :], axis=1)
    arr_mirrored = np.concatenate((arr_flip, arr), axis=1)
    arr_mirrored = arr_mirrored.T
    return arr_mirrored
