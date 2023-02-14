import pathlib as pl
import pickle

import netCDF4 as nc
import numpy as np
import pandas as pd

from .get_space_time_attributes import get_netcdf_latlons
from .get_space_time_attributes import get_netcdf_dates
from .calculate_coordinate_indices import calculate_latlon_indices
from .calculate_date_indices import calculate_date_indices
from .calculate_mapping_indices import calculate_mapping_indices
from .load_data_netcdf_variable import load_data_netcdf_variable_temporal_flat
from .load_data_netcdf_variable import load_data_netcdf_variable_static_flat


def store_data_netcdf(file: pl.Path,
                      samples: np.ndarray,
                      lons: np.ndarray,
                      lats: np.ndarray,
                      dates: np.ndarray,
                      dir_out: pl.Path,
                      verbose: int = 1) -> None:
    
    if not file.exists():
        raise ValueError("{} does not exist".format(file))

    dataset = nc.Dataset(file)
    map_lons, map_lats = get_netcdf_latlons(dataset=dataset)
    map_dates = get_netcdf_dates(dataset=dataset)
    
    flat_lons = np.array([l for _ in map_lats for l in map_lons])
    flat_lats = np.array([l for l in map_lats for _ in map_lons])

    s_indices = calculate_latlon_indices(to_lats = lats,
                                         to_lons = lons,
                                         from_lats = map_lats,
                                         from_lons = map_lons)
    d_indices = np.zeros(shape = (len(dates)), dtype = np.int64)
    
    x_coordinate_step = np.absolute(map_lons[1] - map_lons[0])
    x_resolution = "{:02d}-arcminute".format(round(x_coordinate_step * 60))
    y_coordinate_step = np.absolute(map_lats[1] - map_lats[0])
    y_resolution = "{:02d}-arcminute".format(round(y_coordinate_step * 60))
    d_frequency = "single-year_yearly"
    
    if map_dates is not None:
        d_indices, d_frequency = calculate_date_indices(to_datetime=dates,
                                                        from_datetime=map_dates)
        
    data_s_indices = pd.unique(s_indices)
    data_d_indices = pd.unique(d_indices)
    data_d_frequency = d_frequency
        
    s_mapping = calculate_mapping_indices(from_indices = s_indices,
                                          to_indices = data_s_indices)
    d_mapping = calculate_mapping_indices(from_indices = d_indices,
                                          to_indices = data_d_indices)

    variables = [variable
                 for variable in dataset.variables.values()
                 if variable.name not in ["lat", "lon", "latitude", "longitude", "time"]]
    
    for variable in variables:

        values_out = pl.Path("{}/{}.npy".format(dir_out.parent, variable.name))
        meta_out = pl.Path("{}/{}_meta.pkl".format(dir_out.parent, variable.name))
        if values_out.exists() and meta_out.exists():
            continue
        
        if map_dates is not None:
            values = load_data_netcdf_variable_temporal_flat(variable=variable,
                                                             s_indices=data_s_indices,
                                                             d_indices=data_d_indices)
        else:
            values = load_data_netcdf_variable_static_flat(variable=variable,
                                                           s_indices=data_s_indices)

        data_lons = flat_lons[data_s_indices]
        data_lats = flat_lats[data_s_indices]
        data_dates = dates[data_d_indices]
        
        if map_dates is not None:
            data_dates = map_dates[data_d_indices]

        meta = {"samples": samples,
                "lons": lons,
                "lats": lats,
                "dates": dates,
                "origional_lons": data_lons,
                "origional_lats": data_lats,
                "origional_dates": data_dates,
                "spatial_mapping": s_mapping,
                "dates_mapping": d_mapping,
                "x_resolution": x_resolution,
                "y_resolution": y_resolution,
                "date_frequency": data_d_frequency}
        
        if verbose > 0:
            print("Loaded NetCDF {}".format(values.shape), flush=True)

        values_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(file=values_out, arr=values)
        
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_out, 'wb') as save_file:
            pickle.dump(meta, save_file)

    dataset.close()
