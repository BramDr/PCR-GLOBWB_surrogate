from typing import Optional
import pathlib as pl
import time

import numpy as np
import netCDF4 as nc
import pcraster as pcr

from .aggregate_array import aggregate_array
from .convert_array import convert_array
from .regrid_array_uniform import regrid_array_uniform


def get_nc_array(file: pl.Path,
                 variable_name: str,
                 date_index: Optional[int] = None,
                 aggregate: str = "",
                 aggregate_ids: Optional[pcr.Field] = None,
                 conversion: str = "",
                 area: Optional[pcr.Field] = None) -> np.ndarray:
    
    values = get_nc_array_spatial(file = file,
                                  variable_name=variable_name,
                                  date_index=date_index,
                                  aggregate=aggregate,
                                  aggregate_ids=aggregate_ids,
                                  conversion=conversion,
                                  area=area).flatten()
    return values


def get_nc_array_spatial(file: pl.Path,
                        variable_name: str,
                        date_index: Optional[int] = None,
                        aggregate: str = "",
                        aggregate_ids: Optional[pcr.Field] = None,
                        conversion: str = "",
                        area: Optional[pcr.Field] = None) -> np.ndarray:
    
    # counter_start = time.perf_counter()
    dataset = nc.Dataset(file)
    if variable_name == "":
        variable_name = [name for name in dataset.variables.keys() if name not in ["lat", "lon", "time", "latitude", "longitude", "Lat", "Lon", "Time", "Latitude", "Longitude"]][0]
    
    if date_index is None:
        values = dataset.variables[variable_name][...]
    else:
        values = dataset.variables[variable_name][date_index, ...]
    dataset.close()
    # counter_end = time.perf_counter()
    # print(f"Finished dataset.variables in {counter_end - counter_start} seconds")

    values = values.filled(fill_value=np.nan)
    
    if (conversion != "" and area is not None) or (aggregate != "" and aggregate_ids is not None):
        ncols, nrows = pcr.clone().nrCols(), pcr.clone().nrRows()
        
        factor = 1
        if values.shape != (nrows, ncols):
            factor = int(nrows / values.shape[0])
            
        if factor != 1 and area is not None:
            print("Clone shape is {},{} while values shape is {}".format(nrows, ncols, values.shape))
            raise ValueError("Cannot disaggregate and convert at the same time")
        
        # counter_start = time.perf_counter()
        values = regrid_array_uniform(array=values,
                                      factor=factor,
                                      aggregation_type="disaggregate")
        # counter_end = time.perf_counter()
        # print(f"Finished regrid_array_uniform in {counter_end - counter_start} seconds")
        
        values_map = pcr.numpy2pcr(pcr.Scalar, values, np.nan)
        if conversion != "" and area is not None:
            # counter_start = time.perf_counter()
            values_map = convert_array(array_map = values_map,
                                       conversion = conversion,
                                       area = area)
            # counter_end = time.perf_counter()
            # print(f"Finished convert_array in {counter_end - counter_start} seconds")
            
        if aggregate != "" and aggregate_ids is not None:
            # counter_start = time.perf_counter()
            values_map = aggregate_array(array_map = values_map,
                                         aggregate = aggregate,
                                         ids = aggregate_ids)
            # counter_end = time.perf_counter()
            # print(f"Finished aggregate_array in {counter_end - counter_start} seconds")
        
        values = pcr.pcr2numpy(values_map, np.nan)
        
        # counter_start = time.perf_counter()
        values = regrid_array_uniform(array=values,
                                      factor=factor,
                                      aggregation_type="aggregate")
        # counter_end = time.perf_counter()
        # print(f"Finished regrid_array_uniform in {counter_end - counter_start} seconds")
        
    return values
