from typing import Optional
import pathlib as pl

import numpy as np
import pcraster as pcr

from .aggregate_array import aggregate_array
from .convert_array import convert_array
from .regrid_array_uniform import regrid_array_uniform


def get_pcr_array(file: pl.Path,
                  aggregate: str = "",
                  aggregate_ids: Optional[pcr.Field] = None) -> np.ndarray:
    
    values = get_pcr_array_spatial(file = file,
                                   aggregate = aggregate,
                                   aggregate_ids = aggregate_ids).flatten()
    return values

def get_pcr_array_spatial(file: pl.Path,
                          aggregate: str = "",
                          aggregate_ids: Optional[pcr.Field] = None,
                          conversion: str = "",
                          area: Optional[pcr.Field] = None) -> np.ndarray:
    
    pcr.setclone(str(file))
    map = pcr.readmap(str(file))
    map = pcr.scalar(map)
    values = pcr.pcr2numpy(map, np.nan)
    
    if area is not None or aggregate_ids is not None:
        ncols, nrows = pcr.clone().nrCols(), pcr.clone().nrRows()
        
        factor = 1
        if values.shape != (nrows, ncols):
            factor = nrows / values.shape[0]
            
        if factor != 1 and area is not None:
            raise ValueError("Cannot disaggregate and convert at the same time")
        
        values = regrid_array_uniform(array=values,
                                      factor=factor,
                                      aggregation_type="disaggregate")
            
        values_map = pcr.numpy2pcr(pcr.Scalar, values, np.nan)
        if area is not None:
            values_map = convert_array(array_map = values_map,
                                       conversion = conversion,
                                       area = area)
        if aggregate_ids is not None:
            values_map = aggregate_array(array_map = values_map,
                                         aggregate = aggregate,
                                         ids = aggregate_ids)
        values = pcr.pcr2numpy(values_map, np.nan)
        
        values = regrid_array_uniform(array=values,
                                      factor=factor,
                                      aggregation_type="aggregate")
    
    return values
