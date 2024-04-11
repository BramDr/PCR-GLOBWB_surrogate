from typing import Optional

import numpy as np


def split_array_meta(array: np.ndarray,
                     split_fraction: float,
                     meta: Optional[dict] = None,
                     seed: int = 19920223,
                     verbose: int = 1) -> tuple[np.ndarray, Optional[dict]]:
    
    nsamples = array.shape[-2]
    split_size = int(nsamples * split_fraction)
    
    np.random.seed(seed = seed)
    split_indices = np.random.choice(np.arange(nsamples),
                                        size = split_size,
                                        replace=False)
    
    array = array[..., split_indices, :]
    
    if meta is not None:
        meta["samples"] = np.array(meta["samples"][split_indices])
        meta["lats"] = np.array(meta["lats"][split_indices])
        meta["lons"] = np.array(meta["lons"][split_indices])

    if verbose > 0:
        print("Split array {} with meta".format(array.shape))
        
    return array, meta