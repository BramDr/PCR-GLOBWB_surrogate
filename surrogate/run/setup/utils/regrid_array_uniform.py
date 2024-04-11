
import numpy as np
import numba as nb

@nb.njit
def regrid_array_uniform(array: np.ndarray,
                         factor: int,
                         aggregation_type: str) -> np.ndarray:
        
    if factor == 1:
        return array
    
    nrows = array.shape[0]
    ncols = array.shape[1]
    
    if aggregation_type == "disaggregate":
        disaggregated_array = np.empty((int(nrows * factor), int(ncols * factor)), dtype = array.dtype)
        for i in range(int(nrows * factor)):
            for j in range(int(ncols * factor)):
                disaggregated_array[i, j] = array[int(i / factor), int(j / factor)]
        array = disaggregated_array
    elif aggregation_type == "aggregate":
        aggregated_array = np.empty((int(nrows / factor), int(ncols / factor)), dtype = array.dtype)
        for i in range(int(nrows / factor)):
            for j in range(int(ncols / factor)):
                aggregated_array[i, j] = np.mean(array[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]) # Slow
                # aggregated_array[i, j] = array[i * factor, j * factor] # Excpect uniform disaggregated array
        array = aggregated_array
    else:
        return None
    
    return array