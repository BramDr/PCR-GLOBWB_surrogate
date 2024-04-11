from typing import Optional

import numpy as np

from .calculate_searchsorted_indices import calculate_searchsorted_indices
from .calculate_date_indices import calculate_date_indices


def calculate_spatial_indices(to_lons: np.ndarray,
                              to_lats: np.ndarray,
                              from_lons: Optional[np.ndarray] = None,
                              from_lats: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:

    lon_indices = np.zeros(shape=(to_lons.size,), dtype=np.int64)
    lat_indices = np.zeros(shape=(to_lats.size,), dtype=np.int64)
    
    if from_lons is not None and from_lats is not None:
        lon_indices = calculate_searchsorted_indices(from_array=from_lons,
                                                     to_array=to_lons)
        lat_indices = calculate_searchsorted_indices(from_array=from_lats,
                                                     to_array=to_lats)

    return lon_indices, lat_indices


def calculate_temporal_indices(to_dates: np.ndarray,
                               from_dates: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:

    indices = np.zeros(shape=(to_dates.size,), dtype=np.int64)
    frequency = "single-year_yearly"
    
    if from_dates is not None and from_dates.size > 1:
        indices, frequency = calculate_date_indices(to_date=to_dates,
                                                    from_date=from_dates)

    return indices, frequency
