import pandas as pd
import numpy as np

from .get_catchment_indices import get_catchment_indices

def get_area_sequence(cells: pd.DataFrame,
                      ldd: pd.DataFrame,
                      subset_size: int = 2000,
                      max_tries: int = 100000) -> list[list[np.ndarray]]:
            
    pit_indices = ldd.index[ldd["pit"]]
        
    pits_sequence = []

    for pit_index in pit_indices:
        full_catchment, pit_sequence = get_catchment_indices(outflow_index = pit_index,
                                                             cells = cells,
                                                             ldd = ldd,
                                                             max_tries = max_tries)

        if not full_catchment:
            print("Pit {} catchment not fully in cells".format(pit_index))
            continue
        
        pit_sequence.reverse()
        pits_sequence.append(pit_sequence)

    return pits_sequence