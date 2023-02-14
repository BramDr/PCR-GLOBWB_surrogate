import numpy as np
import numba as nb


@nb.njit
def indices_from_map_array_flat(map: np.ndarray,
                           s_indices: np.ndarray) -> np.ndarray:

    size = s_indices.size
    
    map = map.reshape((-1,))
    
    selected_data = np.full((size), fill_value=np.nan, dtype=np.float32)
    for index, s in enumerate(s_indices):
        selected_data[index] = map[s]

    return selected_data


@nb.njit
def indices_from_map_array(map: np.ndarray,
                           x_indices: np.ndarray,
                           y_indices: np.ndarray) -> np.ndarray:

    size = x_indices.size

    selected_data = np.full((size), fill_value=np.nan, dtype=np.float32)
    for index, (x, y) in enumerate(zip(x_indices, y_indices)):
        selected_data[index] = map[y, x]

    return selected_data


@nb.njit
def indices_from_array(array: np.ndarray,
                       indices: np.ndarray) -> np.ndarray:

    size = indices.size

    selected_data = np.full((size), fill_value=np.nan, dtype=np.float32)
    for to_index, from_index in enumerate(indices):
        selected_data[to_index] = array[from_index]

    return selected_data


@nb.njit
def indices_from_sample_array(array: np.ndarray,
                              indices: np.ndarray) -> np.ndarray:

    sample_len = array.shape[0]
    date_len = indices.size

    array_expanded = np.full((sample_len, date_len),
                             fill_value=np.nan, dtype=np.float32)
    for index in range(sample_len):
        array_expanded[index, :] = indices_from_array(array=array[index, :],
                                                      indices=indices)

    return array_expanded
