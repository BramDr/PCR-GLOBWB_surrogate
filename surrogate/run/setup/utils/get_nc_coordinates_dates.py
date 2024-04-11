from typing import Optional
import pathlib as pl
import datetime as dt

import numpy as np
import netCDF4 as nc


def get_nc_coordinates_dates(file: pl.Path) -> tuple[np.ndarray,
                                                     np.ndarray,
                                                     Optional[np.ndarray]]:
    
    dataset = nc.Dataset(file)
    
    lon_name = [lon_name
                for lon_name in dataset.variables
                if lon_name in ["lon", "longitude"]]
    lat_name = [lat_name
                for lat_name in dataset.variables
                if lat_name in ["lat", "latitude"]]
    time_name = [time_name
                 for time_name in dataset.variables
                 if time_name in ["time"]]
    
    lons = np.array(dataset.variables[lon_name[0]][:]).astype(np.float32)
    lats = np.array(dataset.variables[lat_name[0]][:]).astype(np.float32)
    
    dates = None
    if len(time_name) > 0:
        time_values = np.array(dataset.variables[time_name[0]][:])
        time_units = dataset.variables[time_name[0]].units
        try:
            time_calendar = dataset.variables[time_name[0]].calendar
        except AttributeError:
            time_calendar = u"standard"
        dates = nc.num2date(times=time_values,
                            units=time_units,
                            calendar=time_calendar)
        dates = np.array([dt.datetime(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond) for date in dates])
        
    dataset.close()
    
    return lons, lats, dates
