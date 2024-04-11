from typing import Sequence, Optional

import numpy as np


def process_domains(nsubsets: int,
                    ndomains: int,
                    ndatetimes: int,
                    origional_lons_domains_subsets: Sequence[Sequence[Optional[np.ndarray]]],
                    origional_lats_domains_subsets: Sequence[Sequence[Optional[np.ndarray]]],
                    s_mapping_domains_subsets: Sequence[Sequence[Optional[np.ndarray]]],
                    arrays_domain_subsets: Sequence[Sequence[Sequence[np.ndarray]]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]:
    
    # Reshape spatial information
    origional_lons_subsets = []
    origional_lats_subsets = []
    s_mapping_subsets = []        
    for subset_index in range(nsubsets):
        
        origional_lons = [o_lons[subset_index] for o_lons in origional_lons_domains_subsets if o_lons[subset_index] is not None]
        origional_lats = [o_lats[subset_index] for o_lats in origional_lats_domains_subsets if o_lats[subset_index] is not None]
        s_mapping = [s_map[subset_index] for s_map in s_mapping_domains_subsets if s_map[subset_index] is not None]
        
        s_mapping_new = []
        prev_s_map_max = 0
        for s_map in s_mapping:
            s_map += prev_s_map_max
            prev_s_map_max = np.max(s_map) + 1
            s_mapping_new.append(s_map)
        s_mapping = s_mapping_new
        
        if len(origional_lons) > 0:
            origional_lons = np.concatenate(origional_lons, axis=0)
        if len(origional_lats) > 0:
            origional_lats = np.concatenate(origional_lats, axis=0)
        if len(s_mapping_new) > 0:
            s_mapping = np.concatenate(s_mapping, axis=0)
        
        origional_lons_subsets.append(origional_lons)
        origional_lats_subsets.append(origional_lats)
        s_mapping_subsets.append(s_mapping)
        
    arrays_subsets = []
    for _ in range(nsubsets):
        time_arrays = []
        for _ in range(ndatetimes):
            time_arrays.append(None)
        arrays_subsets.append(time_arrays)
    
    for subset_index in range(nsubsets):
        for time_index in range(ndatetimes):
            
            array = [array[subset_index][time_index] for array in arrays_domain_subsets if array[subset_index][time_index] is not None]
            if len(array) > 0:
                array = np.concatenate(array, axis=0)
                arrays_subsets[subset_index][time_index] = array
                
    return origional_lons_subsets, origional_lats_subsets, s_mapping_subsets, arrays_subsets