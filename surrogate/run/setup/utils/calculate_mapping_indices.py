import numpy as np
import numba as nb


@nb.njit
def calculate_mapping_indices(from_indices: np.ndarray,
                              to_indices: np.ndarray) -> np.ndarray:
    
    indices_len = from_indices.size

    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)
    for index, from_index in enumerate(from_indices):
        to_index = np.where(to_indices == from_index)[0][0]
        indices[index] = to_index
    
    return indices