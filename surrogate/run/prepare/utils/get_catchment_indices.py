import pandas as pd
import numpy as np


def get_catchment_indices(outflow_index: int,
                          cells: pd.DataFrame,
                          ldd: pd.DataFrame,
                          max_tries: int = 10000) -> tuple[bool, list[np.ndarray]]:
    
    catchment_indices = []
    
    full_catchment = True

    indices = np.array([outflow_index])
    catchment_indices.append(indices)  
    
    iteration = 0      
    for iteration in range(max_tries):
        
        ldd_prev = ldd.loc[indices]
        if ldd_prev.index.size == 0:
            break
        
        indices = np.concatenate(ldd_prev["upstream"].to_numpy())
        if len(indices) == 0:
            break
        
        indices_sel = np.isin(indices, cells.index)
        if indices_sel.sum() != indices_sel.size:
            full_catchment = False
            indices = indices[indices_sel]
        
        catchment_indices.append(indices)
            
    if iteration == max_tries - 1:
        raise ValueError("iteration reached max tries")
    
    return full_catchment, catchment_indices
    