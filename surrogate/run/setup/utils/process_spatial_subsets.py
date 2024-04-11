from typing import Sequence, Optional

import numpy as np

from .sort_flip import sort_flip, unflip_array_indices
from .calculate_searchsorted_indices import calculate_searchsorted_indices
from .calculate_spatiotemporal_indices import calculate_spatial_indices
from .calculate_indices_mapping import calculate_indices_mapping


def process_spatial_subset(lons: np.ndarray,
                           lats: np.ndarray,
                           unique_lons: np.ndarray,
                           unique_lats: np.ndarray,
                           unique_lon_indices: np.ndarray,
                           unique_lat_indices: np.ndarray,
                           array_lons: Optional[np.ndarray] = None,
                           array_lats: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    
    lon_unique_indices = calculate_searchsorted_indices(to_array=lons,
                                                        from_array=unique_lons)
    lat_unique_indices = calculate_searchsorted_indices(to_array=lats,
                                                        from_array=unique_lats)
    
    full_lon_indices = unique_lon_indices[lon_unique_indices]
    full_lat_indices = unique_lat_indices[lat_unique_indices]
    
    lons_size = 0
    if array_lons is not None:
        lons_size = array_lons.size
    full_indices = full_lon_indices + full_lat_indices * lons_size
    
    indices, mapping = calculate_indices_mapping(indices = full_indices)
    
    origional_lons = None
    origional_lats = None
    if array_lats is not None and array_lons is not None:
        origional_lons = array_lons[full_lon_indices]
        origional_lats = array_lats[full_lat_indices]
    
    return full_indices, indices, mapping, origional_lons, origional_lats


def process_spatial_subsets(lons_subsets: Sequence[np.ndarray],
                            lats_subsets: Sequence[np.ndarray],
                            array_lons: Optional[np.ndarray] = None,
                            array_lats: Optional[np.ndarray] = None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    
    # Flip lons and lats if they are not sorted so we can use np.searchsorted
    lons_flipped = False
    lats_flipped = False
    if array_lons is not None:
        array_lons, lons_flipped = sort_flip(array=array_lons)
    if array_lats is not None:
        array_lats, lats_flipped = sort_flip(array=array_lats)
    
    # Find the indices of the unique lons and lats
    unique_lons = np.unique(np.concatenate(lons_subsets))
    unique_lats = np.unique(np.concatenate(lats_subsets))
    unique_lon_indices, unique_lat_indices = calculate_spatial_indices(to_lons=unique_lons,
                                                                       to_lats=unique_lats,
                                                                       from_lons=array_lons,
                                                                       from_lats=array_lats)
    
    # Unflip lons and lats, and their indices, if they were flipped
    if lons_flipped:
        unique_lon_indices, array_lons = unflip_array_indices(indices=unique_lon_indices,
                                                              array=array_lons)
    if lats_flipped:
        unique_lat_indices, array_lats = unflip_array_indices(indices=unique_lat_indices,
                                                              array=array_lats)
    
    full_indices_subsets = []
    indices_subsets = []
    mapping_subsets = []
    origional_lats_subsets = []
    origional_lons_subsets = []
    for lons, lats in zip(lons_subsets,
                          lats_subsets):
        
        # Map the unqiue indices to the subset lons and lats
        full_indices, indices, mapping, origional_lons, origional_lats = process_spatial_subset(lons=lons,
                                                                                                lats=lats,
                                                                                                unique_lons=unique_lons,
                                                                                                unique_lats=unique_lats,
                                                                                                unique_lon_indices=unique_lon_indices,
                                                                                                unique_lat_indices=unique_lat_indices,
                                                                                                array_lons=array_lons,
                                                                                                array_lats=array_lats)
        
        full_indices_subsets.append(full_indices)
        indices_subsets.append(indices)
        mapping_subsets.append(mapping)
        origional_lats_subsets.append(origional_lats)
        origional_lons_subsets.append(origional_lons)
    
    return full_indices_subsets, indices_subsets, mapping_subsets, origional_lons_subsets, origional_lats_subsets