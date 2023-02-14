import numpy as np
import numba as nb


@nb.njit
def calculate_latlon_indices(to_lats: np.ndarray,
                             to_lons: np.ndarray,
                             from_lats: np.ndarray,
                             from_lons: np.ndarray) -> np.ndarray:
    
    lat_indices = calculate_coordinate_indices(to_coordinates = to_lats,
                                               from_coordinates = from_lats)
    lon_indices = calculate_coordinate_indices(to_coordinates = to_lons,
                                               from_coordinates = from_lons)
    
    indices_len = lat_indices.size

    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)
    for to_index, (lat_index, lon_index) in enumerate(zip(lat_indices, lon_indices)):

        from_index = lon_index + lat_index * from_lons.size
        indices[to_index] = from_index
        
    return indices


@nb.njit
def calculate_coordinate_indices(to_coordinates: np.ndarray,
                                 from_coordinates: np.ndarray) -> np.ndarray:
    
    indices_len = to_coordinates.size

    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)
    for to_index, to_coordinate in enumerate(to_coordinates):

        diff = np.absolute(from_coordinates - to_coordinate)
        from_index = np.where(diff == diff.min())[0][0]
        indices[to_index] = from_index
        
    return indices
