from typing import Sequence, Optional
import pathlib as pl
import pickle

import numpy as np

def make_meta(samples:np.ndarray,
            lons:np.ndarray,
            lats:np.ndarray,
            s_mapping:np.ndarray,
            origional_lons:Optional[np.ndarray] = None,
            origional_lats:Optional[np.ndarray] = None,
            dates:Optional[np.ndarray] = None,
            origional_dates:Optional[np.ndarray] = None,
            d_mapping:Optional[np.ndarray] = None) -> dict:

    meta = {"samples": samples,
            "lons": lons,
            "lats": lats,
            "origional_lons": origional_lons,
            "origional_lats": origional_lats,
            "spatial_mapping": s_mapping,
            "dates": dates,
            "origional_dates": origional_dates,
            "date_mapping": d_mapping}
    
    return meta

def make_array(arrays: Sequence[Optional[np.ndarray]]) -> np.ndarray:

    arrays = [array for array in arrays if array is not None]
    
    array = np.array([])
    if len(arrays) > 0:
        array = np.stack(arrays, axis=0)
    
    return array