import numpy as np
import numba as nb


@nb.njit
def is_sorted(array: np.ndarray) -> bool:
    return np.all(array[:-1] <= array[1:])


@nb.njit
def sort_flip(array: np.ndarray) -> tuple[np.ndarray, bool]:
    if not is_sorted(array=array):
        array = np.flip(array)
        if not is_sorted(array):
            raise ValueError("array cannot be sorted by flipping")
        return array, True
    return array, False


@nb.njit
def unflip_array_indices(indices: np.ndarray,
                         array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.flip(array)
    indices = array.size - indices - 1
    return indices, array