from typing import Optional

import numpy as np
import pandas as pd

def subset_upstream_array(array: np.ndarray,
                          array_samples: np.ndarray,
                          subset_samples: np.ndarray,
                          ldd: pd.DataFrame,
                          verbose: int = 1) -> Optional[np.ndarray]:

    subset_arrays = []

    for subset_sample in subset_samples:
        samples_sel = ldd['downstream'] == subset_sample
        if(samples_sel.sum() <= 0):
            continue
        samples = ldd.index[samples_sel]
        
        #print("subset sample {} with upsteam samples {}".format(subset_sample, samples))
        
        subset_sel = np.isin(array_samples, samples)
        if np.sum(subset_sel) != len(samples):
            raise ValueError("For subset sample {} not all upstream samples {} are in the array".format(subset_sample,
                                                                                                        samples))
        
        subset_array = array[subset_sel, ...]
        subset_array = np.sum(subset_array, axis = 0)
        subset_arrays.append(subset_array)

    subset_array = None
    if len(subset_arrays) > 0:
        subset_array = np.stack(subset_arrays, axis = 0)

        if verbose > 0:
            print("Subset upstream array: {}".format(subset_array.shape), flush=True)

    return subset_array