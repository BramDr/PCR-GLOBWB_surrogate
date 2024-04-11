from typing import Sequence, Optional
import datetime as dt

import numpy as np

from .sort_flip import sort_flip, unflip_array_indices
from .calculate_searchsorted_indices import calculate_searchsorted_indices
from .calculate_spatiotemporal_indices import calculate_temporal_indices
from .calculate_indices_mapping import calculate_indices_mapping


def process_dates_subset(dates: np.ndarray,
                         unique_dates: np.ndarray,
                         unique_date_indices: np.ndarray,
                         array_dates: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    
    # Convert dates to date values so we can use np.searchsorted
    dates = np.array([date.toordinal() for date in dates])
    unique_dates = np.array([date.toordinal() for date in unique_dates])
        
    date_unique_indices = calculate_searchsorted_indices(to_array=dates,
                                                         from_array=unique_dates)
    
    # Convert date values to dates so we can continue
    dates = np.array([dt.date.fromordinal(date) for date in dates])
    unique_dates = np.array([dt.date.fromordinal(date) for date in unique_dates])
    
    full_indices = unique_date_indices[date_unique_indices]
    
    indices, mapping = calculate_indices_mapping(indices = full_indices)

    origional_dates = None
    if array_dates is not None:
        origional_dates = array_dates[full_indices]
        
    return full_indices, indices, mapping, origional_dates


def process_dates_subsets(dates_subsets: Sequence[np.ndarray],
                          array_dates: Optional[np.ndarray] = None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[Optional[np.ndarray]]]:
    
    
    # Flip dates if they are not sorted so we can use np.searchsorted
    dates_flipped = False
    if array_dates is not None:
        # Convert dates to date values so we can use is_sorted
        array_dates = np.array([date.toordinal() for date in array_dates])
        
        array_dates, dates_flipped = sort_flip(array=array_dates)
        
        # Convert date values to dates so we can continue
        array_dates = np.array([dt.date.fromordinal(date) for date in array_dates])
        
    
    # Find the indices of the unique dates
    unique_dates = np.unique(np.concatenate(dates_subsets))
    unique_date_indices, frequency = calculate_temporal_indices(to_dates=unique_dates,
                                                                from_dates=array_dates)
    
    # Unflip dates, and their indices, if they were flipped
    if dates_flipped:
        unique_date_indices, array_dates = unflip_array_indices(indices=unique_date_indices,
                                                                array=array_dates)
    
    origional_dates_subsets = []
    full_indices_subsets = []
    indices_subsets = []
    mapping_subsets = []
    frequency_subsets = []
    for dates in dates_subsets:

        # Map the unqiue indices to the subset dates
        full_indices, indices, mapping, origional_dates = process_dates_subset(dates=dates,
                                                                               unique_dates=unique_dates,
                                                                               unique_date_indices=unique_date_indices,
                                                                               array_dates=array_dates)
        
        full_indices_subsets.append(full_indices)
        indices_subsets.append(indices)
        mapping_subsets.append(mapping)
        frequency_subsets.append(frequency)
        origional_dates_subsets.append(origional_dates)
    
    return full_indices_subsets, indices_subsets, mapping_subsets, frequency_subsets, origional_dates_subsets