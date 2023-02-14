from typing import Optional

import numpy as np


def constant_meta_array(constant: float,
                        samples: np.ndarray,
                        lons: np.ndarray,
                        lats: np.ndarray,
                        resolution: str,
                        dates: np.ndarray,
                        verbose: int = 1) -> tuple[dict, Optional[np.ndarray]]:

    s_indices = np.zeros(shape = (len(lons)), dtype = np.int64)
    d_indices = np.zeros(shape = (len(dates)), dtype = np.int64)
    d_frequency = "single-year_yearly"

    meta = {"samples": samples,
            "lons": lons,
            "lats": lats,
            "dates": dates,
            "origional_lons": lons[0],
            "origional_lats": lats[0],
            "origional_dates": dates[0],
            "spatial_mapping": s_indices,
            "dates_mapping": d_indices,
            "x_resolution": resolution,
            "y_resolution": resolution,
            "date_frequency": d_frequency}
    
    if verbose > 0:
        print("Loaded meta", flush=True)
        
    samples_len = 1
    dates_len = 1
    features_len = 1
    
    array = np.full(shape=(samples_len, dates_len, features_len), fill_value=constant, dtype=np.float32)
    
    if verbose > 0:
        print("Loaded constant array {}".format(array.shape), flush=True)
        
    return meta, array