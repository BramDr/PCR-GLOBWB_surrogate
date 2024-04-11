from typing import Sequence, Optional
import pathlib as pl
import pickle

import numpy as np

from .make_array_meta import make_meta
from .make_array_meta import make_array


def store_subsets(arrays_subsets: Sequence[Sequence[Optional[np.ndarray]]],
                    process_flag_subsets: Sequence[bool],
                    meta_out_subsets: Sequence[pl.Path],
                    array_out_subsets: Sequence[pl.Path],
                    samples_subsets: Sequence[np.ndarray],
                    lons_subsets: Sequence[np.ndarray],
                    lats_subsets: Sequence[np.ndarray],
                    s_mapping_subsets: Sequence[np.ndarray],
                    origional_lons_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                    origional_lats_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                    dates_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                    origional_dates_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                    d_mapping_subsets: Optional[Sequence[Optional[np.ndarray]]] = None) -> None:
    
    if origional_lons_subsets is None:
        origional_lons_subsets = [None for _ in range(len(samples_subsets))]
    if origional_lats_subsets is None:
        origional_lats_subsets = [None for _ in range(len(samples_subsets))]
        
    if dates_subsets is None:
        dates_subsets = [None for _ in range(len(samples_subsets))]
    if origional_dates_subsets is None:
        origional_dates_subsets = [None for _ in range(len(samples_subsets))]
    if d_mapping_subsets is None:
        d_mapping_subsets = [None for _ in range(len(samples_subsets))]
    
    for subset_index, process_flag in enumerate(process_flag_subsets):
        meta_out = meta_out_subsets[subset_index]
        samples = samples_subsets[subset_index]
        lons = lons_subsets[subset_index]
        lats = lats_subsets[subset_index]
        s_mapping = s_mapping_subsets[subset_index]
        origional_lons = origional_lons_subsets[subset_index]
        origional_lats = origional_lats_subsets[subset_index]
        dates = dates_subsets[subset_index]
        d_mapping = d_mapping_subsets[subset_index]
        origional_dates = origional_dates_subsets[subset_index]
        
        if not process_flag:
            continue
        
        meta = make_meta(samples = samples,
                        lons = lons,
                        lats = lats,
                        s_mapping = s_mapping,
                        origional_lons = origional_lons,
                        origional_lats = origional_lats,
                        dates = dates,
                        origional_dates = origional_dates,
                        d_mapping = d_mapping)

        meta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_out, "wb") as handle:
            pickle.dump(meta, handle)
        
    for subset_index, process_flag in enumerate(process_flag_subsets):
        array_out = array_out_subsets[subset_index]
        arrays = arrays_subsets[subset_index]
        
        if not process_flag:
            continue
        
        array = make_array(arrays = arrays)

        array_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_out, array)
