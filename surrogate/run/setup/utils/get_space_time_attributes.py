from typing import Optional
import pathlib as pl

import pcraster as pcr
import numpy as np
import netCDF4 as nc


def get_map_latlons(file: pl.Path) -> tuple[np.ndarray, np.ndarray]:
    map = pcr.readmap(str(file))
    map = pcr.defined(map) | ~pcr.defined(map)

    map_x = pcr.xcoordinate(map)
    map_y = pcr.ycoordinate(map)
    lons = pcr.pcr2numpy(map_x, np.nan)
    lats = pcr.pcr2numpy(map_y, np.nan)

    lons = np.median(lons, axis=0)
    lats = np.median(lats, axis=1)
    
    return lons, lats


def get_netcdf_latlons(dataset: nc.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lon_name = [lon_name
                for lon_name in dataset.variables
                if lon_name in ["lon", "longitude"]]
    lat_name = [lat_name
                for lat_name in dataset.variables
                if lat_name in ["lat", "latitude"]]

    lons = dataset.variables[lon_name[0]][:]
    lats = dataset.variables[lat_name[0]][:]
        
    return lons, lats


def get_netcdf_dates(dataset: nc.Dataset) -> Optional[np.ndarray]:
    time_name = [time_name
                 for time_name in dataset.variables
                 if time_name in ["time"]]

    time = None
    if len(time_name) > 0:
        time_values = dataset.variables[time_name[0]][:]
        time_units = dataset.variables[time_name[0]].units

        try:
            time_calendar = dataset.variables[time_name[0]].calendar
        except AttributeError:
            time_calendar = u"standard"

        time = nc.num2date(times=time_values,
                           units=time_units,
                           calendar=time_calendar)
        time = np.array(time)

    return time
