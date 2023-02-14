import pathlib as pl

import numpy as np

from .store_data_netcdf import store_data_netcdf


def store_data_output_directory(samples: np.ndarray,
                                lons: np.ndarray,
                                lats: np.ndarray,
                                dates: np.ndarray,
                                output_dir: pl.Path,
                                dir_out: pl.Path = pl.Path("."),
                                verbose: int = 1):

    if verbose > 0:
        print("Working on output directory {}".format(output_dir), flush=True)

    if not output_dir.is_dir():
        raise ValueError("{} is not a directory".format(output_dir))

    output_files = [file for file in output_dir.iterdir() if not file.is_dir()]

    for output_file in output_files:
        
        variable = output_file.name.split("_")[0]
        variable_out = "{}/{}".format(dir_out, variable)
        variable_out = pl.Path(variable_out)

        print("Output {} - variable {}".format(output_file.stem, variable))

        store_data_netcdf(file=output_file,
                          samples=samples,
                          lons=lons,
                          lats=lats,
                          dates=dates,
                          dir_out=variable_out)
