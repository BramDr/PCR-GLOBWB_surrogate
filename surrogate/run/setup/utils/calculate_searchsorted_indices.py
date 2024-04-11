import numpy as np
import numba as nb

@nb.njit
def calculate_searchsorted_indices(from_array: np.ndarray,
                                   to_array: np.ndarray) -> np.ndarray:
    
    indices = np.searchsorted(from_array, to_array)
    indices = np.clip(indices, 1, from_array.size - 1)
    
    left_coordinate = from_array[indices-1]
    right_coordinate = from_array[indices]
    
    indices -= to_array - left_coordinate < right_coordinate - to_array
    return indices