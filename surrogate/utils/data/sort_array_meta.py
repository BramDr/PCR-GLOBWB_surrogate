import numpy as np

def sort_array_meta(array: np.ndarray,
                    meta: dict,
                    verbose: int = 1) -> tuple[np.ndarray, dict]:

    sort_indices = np.argsort(meta["samples"])
    meta["samples"] = meta["samples"][sort_indices]
    meta["lons"] = meta["lons"][sort_indices]
    meta["lats"] = meta["lats"][sort_indices]
    array = array[:, sort_indices, :]

    if verbose > 0:
        print("Sorted array: {} with meta".format(array.shape))
        
    return array, meta
