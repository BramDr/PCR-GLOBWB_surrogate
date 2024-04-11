import pandas as pd
import numpy as np

from .get_catchment_indices import get_catchment_indices


def get_point_sequence(ldd: pd.DataFrame,
                       cells: pd.DataFrame,
                       point: list,
                       subset_size: int = 1000,
                       max_tries: int = 100000) -> list[np.ndarray]:
    
    lon_diff = np.abs(cells["lon"] - point[0])
    lat_diff = np.abs(cells["lat"] - point[1])
    lon_sel = lon_diff == np.min(lon_diff)
    lat_sel = lat_diff == np.min(lat_diff)
    point_sel = np.logical_and(lon_sel, lat_sel)
    point_index = cells.index[point_sel][0]
    
    pit_index = point_index
    for _ in range(max_tries):
        downstream_index = ldd.at[pit_index, "downstream"]
        if ldd.at[downstream_index, "pit"]:
            break
        pit_index = downstream_index
    
    full_catchment, pit_sequence = get_catchment_indices(outflow_index = pit_index,
                                                         cells = cells,
                                                         ldd = ldd,
                                                         max_tries = max_tries)
    
    if not full_catchment:
        raise ValueError("Pit {} catchment not fully in cells".format(pit_index))
    
    pit_sequence.reverse()
            
    return pit_sequence