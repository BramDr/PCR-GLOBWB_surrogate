import pathlib as pl

import pandas as pd

from utils.store_data_input_configuration import store_data_input_configuration
from utils.store_data_output_directory import store_data_output_directory

feature_dir = pl.Path("../features/saves/global_05min")
configuration_dir = pl.Path("../../../PCR-GLOBWB/configuration/subset")
output_dir = pl.Path("../../../PCR-GLOBWB/output/global_05min/subsets")
save_dir = pl.Path("./saves/global_05min")
dir_out = pl.Path("./saves/global_05min")

submasks = [dir.stem for dir in feature_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_code = submask.split("_")[1]
    
    feature_submask_dir = pl.Path("{}/{}".format(feature_dir,submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))
    
    configuration_submask_dir = pl.Path("{}/{}".format(configuration_dir, submask_code))
    output_submask_dir = pl.Path("{}/{}/netcdf".format(output_dir, submask_code))
        
    cells_file = pl.Path("{}/cells.csv".format(feature_submask_dir))
    cells = pd.read_csv(cells_file, index_col=0)
    
    samples = cells.index.to_numpy()
    lons = cells["lon"].to_numpy()
    lats = cells["lat"].to_numpy()
    dates = pd.date_range(start=cells["start"].iloc[0],
                        end=cells["end"].iloc[0],
                        freq="D").to_pydatetime()

    input_out = pl.Path("{}/input".format(submask_out))
    output_out = pl.Path("{}/output".format(submask_out))
    
    configuration_file = pl.Path("{}/global_05min.ini".format(configuration_submask_dir))
    store_data_input_configuration(samples = samples,
                                    lons = lons,
                                    lats = lats,
                                    dates = dates,
                                    configuration_file=configuration_file,
                                    dir_out=input_out)

    store_data_output_directory(samples = samples,
                                lons = lons,
                                lats = lats,
                                dates = dates,
                                output_dir=output_submask_dir,
                                dir_out=output_out)
