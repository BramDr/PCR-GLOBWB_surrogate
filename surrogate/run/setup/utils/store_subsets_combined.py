from typing import Sequence, Optional
import pathlib as pl
import pickle

import numpy as np

from .make_array_meta import make_meta
from .make_array_meta import make_array


def store_subsets_combined(names: Sequence[str],
                           arrays_subsets: Sequence[Sequence[Optional[np.ndarray]]],
                           process_flag_subsets: Sequence[bool],
                           meta_out_subsets: Sequence[pl.Path],
                           array_out_subsets: Sequence[pl.Path],
                           samples_subsets: Sequence[np.ndarray],
                           lons_subsets: Sequence[np.ndarray],
                           lats_subsets: Sequence[np.ndarray],
                           s_mapping_subsets: Sequence[np.ndarray],
                           arrays_additional: Optional[dict[str, np.ndarray]] = None,
                           origional_lons_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                           origional_lats_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                           dates_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                           origional_dates_subsets: Optional[Sequence[Optional[np.ndarray]]] = None,
                           d_mapping_subsets: Optional[Sequence[Optional[np.ndarray]]] = None) -> None:
    
    if arrays_additional is None:
        arrays_additional = {}
    if origional_lons_subsets is None:
        origional_lons_subsets_sel = [None for _ in range(len(process_flag_subsets))]
    if origional_lats_subsets is None:
        origional_lats_subsets_sel = [None for _ in range(len(process_flag_subsets))]
    if dates_subsets is None:
        dates_subsets_sel = [None for _ in range(len(process_flag_subsets))]
    if origional_dates_subsets is None:
        origional_dates_subsets_sel = [None for _ in range(len(process_flag_subsets))]
    if d_mapping_subsets is None:
        d_mapping_subsets_sel =[None for _ in range(len(process_flag_subsets))]
    
    for array_out_subset in np.unique(array_out_subsets):
        subset_indices = [i for i, out in enumerate(array_out_subsets) if out == array_out_subset]
        
        names_sel = [names[i] for i in subset_indices]
        arrays_subsets_sel = [arrays_subsets[i] for i in subset_indices]
        process_flag_subsets_sel = [process_flag_subsets[i] for i in subset_indices]
        meta_out_subsets_sel = [meta_out_subsets[i] for i in subset_indices]
        array_out_subsets_sel = [array_out_subsets[i] for i in subset_indices]
        samples_subsets_sel = [samples_subsets[i] for i in subset_indices]
        lons_subsets_sel = [lons_subsets[i] for i in subset_indices]
        lats_subsets_sel = [lats_subsets[i] for i in subset_indices]
        s_mapping_subsets_sel = [s_mapping_subsets[i] for i in subset_indices]
        origional_lons_subsets_sel = [origional_lons_subsets[i] for i in subset_indices]
        origional_lats_subsets_sel = [origional_lats_subsets[i] for i in subset_indices]
        dates_subsets_sel = [dates_subsets[i] for i in subset_indices]
        origional_dates_subsets_sel = [origional_dates_subsets[i] for i in subset_indices]
        d_mapping_subsets_sel = [d_mapping_subsets[i] for i in subset_indices]
        
        make_list = len(names_sel) != np.unique(names_sel).size
        
        metas_dict = {}
        for subset_index, process_flag in enumerate(process_flag_subsets_sel):
            name = names_sel[subset_index]
            samples = samples_subsets_sel[subset_index]
            lons = lons_subsets_sel[subset_index]
            lats = lats_subsets_sel[subset_index]
            s_mapping = s_mapping_subsets_sel[subset_index]
            origional_lons = origional_lons_subsets_sel[subset_index]
            origional_lats = origional_lats_subsets_sel[subset_index]
            
            if not process_flag:
                continue
            
            meta = make_meta(samples = samples,
                             lons = lons,
                             lats = lats,
                             s_mapping = s_mapping,
                             origional_lons = origional_lons,
                             origional_lats = origional_lats)
            
            if make_list:
                if name not in metas_dict.keys():
                    metas_dict[name] = []
                metas_dict[name].append(meta)
            else:
                metas_dict[name] = meta
            
        metas_dict["dates"] = dates_subsets_sel[0]
        metas_dict["date_mapping"] = d_mapping_subsets_sel[0]
        metas_dict["origional_dates"] = origional_dates_subsets_sel[0]
    
        arrays_dict = {}
        for subset_index, process_flag in enumerate(process_flag_subsets_sel):
            name = names_sel[subset_index]
            arrays = arrays_subsets_sel[subset_index]
            
            if not process_flag:
                continue
            
            array = make_array(arrays = arrays)
            
            if make_list:
                if name not in arrays_dict.keys():
                    arrays_dict[name] = []
                arrays_dict[name].append(array)
            else:
                arrays_dict[name] = array
            
        if make_list:
            for name, arrays in arrays_dict.items():
                array = np.empty(len(arrays), dtype = "object")
                array[:] = arrays
                arrays_dict[name] = array

        array_out = array_out_subsets_sel[0]
        array_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_out, **arrays_dict, **arrays_additional)
        
        meta_out = meta_out_subsets_sel[0]
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_out, "wb") as handle:
            pickle.dump(metas_dict, handle)
