import pathlib as pl

import pandas as pd

from utils.store_data_input_configuration import store_data_input_configuration
from utils.store_data_output_directory import store_data_output_directory

configuration_file = pl.Path("../../../PCR-GLOBWB/configuration/global_30min.ini")
output_dir = pl.Path("../../../PCR-GLOBWB/output/global_30min/netcdf")
feature_dir = pl.Path("../features/saves/global_30min")
input_out = pl.Path("./saves/global_30min/input")
output_out = pl.Path("./saves/global_30min/output")

cells_file = pl.Path("{}/cells.csv".format(feature_dir))
cells = pd.read_csv(cells_file, index_col=0)

samples = cells.index.to_numpy()
lons = cells["lon"].to_numpy()
lats = cells["lat"].to_numpy()
dates = pd.date_range(start=cells["start"].iloc[0],
                      end=cells["end"].iloc[0],
                      freq="D").to_pydatetime()

#store_data_input_configuration(samples = samples,
#                                lons = lons,
#                                lats = lats,
#                                dates = dates,
#                                configuration_file=configuration_file,
#                                dir_out=input_out)

store_data_output_directory(samples = samples,
                            lons = lons,
                            lats = lats,
                            dates = dates,
                            output_dir=output_dir,
                            dir_out=output_out)
