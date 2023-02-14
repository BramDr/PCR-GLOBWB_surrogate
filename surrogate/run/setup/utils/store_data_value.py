import pathlib as pl

import numpy as np

from .store_data_netcdf import store_data_netcdf
from .store_data_pcraster import store_data_pcraster


def store_data_value(value: pl.Path,
                     samples: np.ndarray,
                     lons: np.ndarray,
                     lats: np.ndarray,
                     dates: np.ndarray,
                     dir_out: pl.Path,
                     prefix: pl.Path) -> None:

    if value.suffix in [".map"]:
        file = "{}/{}".format(prefix, value)
        file = pl.Path(file)
        
        store_data_pcraster(file=file,
                            samples=samples,
                            lons=lons,
                            lats=lats,
                            dates=dates,
                            dir_out=dir_out)

    elif value.suffix in [".nc", ".nc4"]:
        file = "{}/{}".format(prefix, value)
        file = pl.Path(file)
        
        store_data_netcdf(file=file,
                          samples=samples,
                          lons=lons,
                          lats=lats,
                          dates=dates,
                          dir_out=dir_out)
