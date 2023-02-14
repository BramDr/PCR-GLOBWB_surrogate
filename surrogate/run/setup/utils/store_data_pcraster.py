import pathlib as pl
import pickle

import numpy as np
import pandas as pd
import pcraster as pcr

from .get_space_time_attributes import get_map_latlons
from .calculate_coordinate_indices import calculate_latlon_indices
from .calculate_mapping_indices import calculate_mapping_indices
from .load_data_pcraster_variable import load_data_pcraster_variable_flat


def store_data_pcraster(file: pl.Path,
                        samples: np.ndarray,
                        lons: np.ndarray,
                        lats: np.ndarray,
                        dates: np.ndarray,
                        dir_out: pl.Path,
                        verbose: int = 1) -> None:

    if not file.exists():
        raise ValueError("{} does not exist".format(file))

    pcr.setclone(str(file))
    map = pcr.readmap(str(file))
    map_lons, map_lats = get_map_latlons(file)
    
    flat_lons = np.array([l for _ in map_lats for l in map_lons])
    flat_lats = np.array([l for l in map_lats for _ in map_lons])
    
    s_indices = calculate_latlon_indices(to_lats = lats,
                                         to_lons = lons,
                                         from_lats = map_lats,
                                         from_lons = map_lons)
    d_indices = np.zeros(shape = (len(dates)), dtype = np.int64)
    
    x_coordinate_step = map_lons[1] - map_lons[0]
    x_resolution = "{:02d}-arcminute".format(round(x_coordinate_step * 60))
    y_coordinate_step = map_lats[1] - map_lats[0]
    y_resolution = "{:02d}-arcminute".format(round(y_coordinate_step * 60))
    d_frequency = "single-year_yearly"
    
    data_s_indices = pd.unique(s_indices)
    data_d_indices = pd.unique(d_indices)
    data_d_frequency = d_frequency

    s_mapping = calculate_mapping_indices(from_indices = s_indices,
                                          to_indices = data_s_indices)
    d_mapping = calculate_mapping_indices(from_indices = d_indices,
                                          to_indices = data_d_indices)
    
    values_out = pl.Path("{}.npy".format(dir_out))
    meta_out = pl.Path("{}_meta.pkl".format(dir_out))
    if values_out.exists() and meta_out.exists():
        return

    values = load_data_pcraster_variable_flat(map=map,
                                              s_indices=data_s_indices)

    variable_lons = flat_lons[data_s_indices]
    variable_lats = flat_lats[data_s_indices]
    variable_dates = dates[data_d_indices]
    
    meta = {"samples": samples,
            "lons": lons,
            "lats": lats,
            "dates": dates,
            "origional_lons": variable_lons,
            "origional_lats": variable_lats,
            "origional_dates": variable_dates,
            "spatial_mapping": s_mapping,
            "dates_mapping": d_mapping,
            "x_resolution": x_resolution,
            "y_resolution": y_resolution,
            "date_frequency": data_d_frequency}

    if verbose > 0:
        print("Loaded PCRaster {}".format(values.shape), flush=True)

    values_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(file=values_out, arr=values)
    
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, 'wb') as save_file:
        pickle.dump(meta, save_file)
