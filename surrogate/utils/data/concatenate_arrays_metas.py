from typing import Sequence, Optional

import numpy as np


def concatenate_arrays(arrays: Sequence[np.ndarray],
                       direction: str) -> np.ndarray:

    if direction == "sample":
        concatenate_axis = -2
    elif direction == "date":
        concatenate_axis = 0
    elif direction == "feature":
        concatenate_axis = -1
    else:
        raise NotImplementedError(
            "Cannot concatinate array direction {}".format(direction))

    concatenated = np.concatenate(arrays, axis=concatenate_axis)
    
    return concatenated


def concatenate_metas(metas: Sequence[dict],
                      direction: str) -> dict:

    concatenated = metas[0].copy()
    if direction == "sample":
        samples = [meta["samples"] for meta in metas]
        lons = [meta["lons"] for meta in metas]
        lats = [meta["lats"] for meta in metas]
        concatenated["samples"] = np.concatenate(samples)
        concatenated["lons"] = np.concatenate(lons)
        concatenated["lats"] = np.concatenate(lats)
        if "origional_lons" in concatenated.keys():
            origional_lons = [meta["origional_lons"] for meta in metas if meta["origional_lons"] is not None]
            if len(origional_lons) > 0:
                concatenated["origional_lons"] = np.concatenate(origional_lons)
        if "origional_lats" in concatenated.keys():
            origional_lats = [meta["origional_lats"] for meta in metas if meta["origional_lats"] is not None]
            if len(origional_lats) > 0:
                concatenated["origional_lats"] = np.concatenate(origional_lats)
    elif direction == "date":
        dates = [meta["dates"] for meta in metas]
        concatenated["dates"] = np.concatenate(dates)
        if "origional_dates" in concatenated.keys():
            origional_dates = [meta["origional_dates"] for meta in metas if meta["origional_dates"] is not None]
            if len(origional_dates) > 0:
                concatenated["origional_dates"] = np.concatenate(origional_dates)
    elif direction == "feature":
        features = [meta["features"] for meta in metas]
        concatenated["features"] = np.concatenate(features)
    
    return concatenated


def concatenate_arrays_metas(arrays: Optional[Sequence[np.ndarray]],
                             metas: Optional[Sequence[dict]],
                             direction: str,
                             verbose: int = 1) -> tuple[Optional[np.ndarray], Optional[dict]]:

    concatenated_array = None
    concatenated_meta = None
    if arrays is not None:
        concatenated_array = concatenate_arrays(arrays=arrays, direction=direction)
    if metas is not None:
        concatenated_meta = concatenate_metas(metas=metas, direction=direction)

    if verbose > 0:
        narrays = 0
        array_shape = None
        if arrays is not None:
            narrays = len(arrays)
            array_shape = concatenated_array.shape
            
        print("Concatinated {} arrays: {} with metas".format(narrays,
                                                             array_shape))

    return concatenated_array, concatenated_meta
