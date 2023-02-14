import numpy as np
        
from surrogate.utils.data import concatenate_arrays_metas

def combine_spatiotemporal_arrays_metas(train: np.ndarray,
                                    train_meta: dict,
                                    spatial_test: np.ndarray,
                                    spatial_test_meta: dict,
                                    temporal_test: np.ndarray,
                                    temporal_test_meta: dict,
                                    spatiotemporal_test: np.ndarray,
                                    spatiotemporal_test_meta: dict,
                                    verbose: int = 1) -> tuple[np.ndarray, dict]:
    
    spatial, spatial_meta = concatenate_arrays_metas(arrays=[train,spatial_test],
                                                     metas=[train_meta,spatial_test_meta],
                                                     direction="sample",
                                                     verbose = verbose-1)
    
    temporal, temporal_meta = concatenate_arrays_metas(arrays=[temporal_test,spatiotemporal_test],
                                                        metas=[temporal_test_meta,spatiotemporal_test_meta],
                                                        direction="sample",
                                                        verbose = verbose-1)
    
    array, meta = concatenate_arrays_metas(arrays=[spatial,temporal],
                                           metas=[spatial_meta,temporal_meta],
                                           direction="date",
                                           verbose = verbose-1)
    
    if verbose > 0:
        print("Combined train and test arrays and metas", flush=True)
        
    return array, meta
