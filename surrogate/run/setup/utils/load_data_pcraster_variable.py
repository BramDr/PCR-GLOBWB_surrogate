import numpy as np
import pcraster as pcr

from .get_data_indices import indices_from_map_array
from .get_data_indices import indices_from_map_array_flat


def load_data_pcraster_variable_flat(map: pcr.Field,
                                     s_indices: np.ndarray) -> np.ndarray:

    values = pcr.pcr2numpy(map=map,
                           mv=np.nan)
    values = indices_from_map_array_flat(map=values,
                                         s_indices=s_indices)

    return values


def load_data_pcraster_variable(map: pcr.Field,
                                x_indices: np.ndarray,
                                y_indices: np.ndarray) -> np.ndarray:

    values = pcr.pcr2numpy(map=map,
                           mv=np.nan)
    values = indices_from_map_array(map=values,
                                    x_indices=x_indices,
                                    y_indices=y_indices)

    return values
