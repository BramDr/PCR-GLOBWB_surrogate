from typing import Optional

import numpy as np
import pandas as pd

def _subset_meta_samples(meta: dict,
                         samples_indices: np.ndarray) -> dict:
    
    meta = meta.copy()
    
    meta["samples"] = meta["samples"][samples_indices]
    meta["lons"] = meta["lons"][samples_indices]
    meta["lats"] = meta["lats"][samples_indices]
    
    s_indices = meta["spatial_mapping"][samples_indices]
    data_s_indices = pd.unique(s_indices)
    s_mapping = np.array([np.where(data_s_indices == index)[0][0] for index in s_indices])

    meta["origional_lons"] = meta["origional_lons"][data_s_indices]
    meta["origional_lats"] = meta["origional_lats"][data_s_indices]
    meta["spatial_mapping"] = s_mapping

    return meta

def _subset_array_samples(array: np.ndarray,
                          meta: dict,
                          samples_indices: np.ndarray) -> np.ndarray:
        
    s_indices = meta["spatial_mapping"][samples_indices]
    data_s_indices = pd.unique(s_indices)
    
    array = array[data_s_indices, :, :]

    return array


def _subset_meta_dates(meta: dict,
                       dates_indices: np.ndarray) -> dict:
    
    meta = meta.copy()
    
    meta["dates"] = meta["dates"][dates_indices]

    d_indices = meta["dates_mapping"][dates_indices]
    data_d_indices = pd.unique(d_indices)
    d_mapping = np.array([np.where(data_d_indices == index)[0][0] for index in d_indices])

    meta["origional_dates"] = meta["origional_dates"][data_d_indices]
    meta["dates_mapping"] = d_mapping

    return meta


def _subset_array_dates(array: np.ndarray,
                        meta: dict,
                        dates_indices: np.ndarray) -> np.ndarray:
    
    d_indices = meta["dates_mapping"][dates_indices]
    data_d_indices = pd.unique(d_indices)
    
    array = array[:, data_d_indices, :]

    return array


def subset_meta_array(meta: dict,
                      array: Optional[np.ndarray] = None,
                      samples: Optional[np.ndarray] = None,
                      dates: Optional[np.ndarray] = None,
                      verbose: int = 1) -> tuple[dict, Optional[np.ndarray]]:

    if samples is not None:
        samples_indices = np.array([np.where(meta["samples"] == sample)[0][0] for sample in samples], dtype=np.int64)
        
        if array is not None:
            array = _subset_array_samples(array = array,
                                          meta=meta,
                                          samples_indices=samples_indices)
        meta = _subset_meta_samples(meta=meta,
                                    samples_indices=samples_indices)
            

    if dates is not None:
        dates_indices = np.array([np.where(meta["dates"] == datum)[0][0] for datum in dates], dtype=np.int64)
        if array is not None:
            array = _subset_array_dates(array = array,
                                    meta=meta,
                                    dates_indices=dates_indices)
        meta = _subset_meta_dates(meta=meta,
                                  dates_indices=dates_indices)

    if array is not None:
        if verbose > 0:
            print("Subset array: {}".format(array.shape), flush=True)

    return meta, array