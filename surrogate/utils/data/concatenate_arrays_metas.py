from typing import Sequence
import copy

import numpy as np


def concatenate_arrays_metas(arrays: Sequence[np.ndarray],
                             metas: Sequence[dict],
                             direction: str = "spatial",
                             verbose: int = 1) -> tuple[np.ndarray, dict]:

    if direction == "sample":
        concatenate_axis = 0
    elif direction == "upstream":
        concatenate_axis = -2
    elif direction == "date":
        concatenate_axis = 1
    elif direction == "feature":
        concatenate_axis = -1
    else:
        raise NotImplementedError("Cannot concatinate array direction {}".format(direction))

    concatenated = np.concatenate(arrays, axis=concatenate_axis)

    concatenated_meta = copy.deepcopy(metas[0])
    if direction == "sample":
        samples = [sample for meta in metas for sample in meta["samples"]]
        lons = [lon for meta in metas for lon in meta["lons"]]
        lats = [lat for meta in metas for lat in meta["lats"]]
        concatenated_meta["samples"] = np.array(samples)
        concatenated_meta["lons"] = np.array(lons)
        concatenated_meta["lats"] = np.array(lats)
    elif direction == "date":
        dates = [datum for meta in metas for datum in meta["dates"]]
        concatenated_meta["dates"] = np.array(dates)
    elif direction == "feature":
        features = [feature for meta in metas for feature in meta["features"]]
        concatenated_meta["features"] = np.array(features)

    if verbose > 0:
        print("Concatinated {} arrays: {} with metas".format(len (arrays),
                                                           concatenated.shape))

    return concatenated, concatenated_meta
