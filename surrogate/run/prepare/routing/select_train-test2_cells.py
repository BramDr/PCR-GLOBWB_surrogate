import pathlib as pl
import datetime as dt

import pandas as pd
import numpy as np
import netCDF4 as nc

save_dir = pl.Path("./saves")
base_dir = pl.Path("../saves")
out_dir = pl.Path("./saves/train-test2")
seed = 19920223
time_start = dt.date(2000, 1, 1)
time_end = dt.date(2010, 12, 31)

routing_details = {"river": {"routing_number": 0,
                               "base_fraction": 1 / 8,
                               "resolution_fractions": {"30min": 1 / 1,
                                                        "05min": 1 / 16,
                                                        "30sec": 1 / 64},},
                     "lake": {"routing_number": 1,
                              "base_fraction": 1 / 4,
                              "resolution_fractions": {"30min": 1 / 1,
                                                       "05min": 1 / 1,
                                                       "30sec": 1 / 1},},
                     "reservoir": {"routing_number": 2,
                                   "base_fraction": 1 / 4,
                                   "resolution_fractions": {"30min": 1 / 1,
                                                            "05min": 1 / 1,
                                                            "30sec": 1 / 1},},}
test_fraction = 1 / 3
validate_fraction = 1 / 4
temporal_fraction = 2 / 3
nbins = 5

time_split = time_start + (time_end - time_start) * temporal_fraction

resolutions = ["30min", "05min", "30sec"]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))

    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
    base_resolution_dir = pl.Path("{}/{}".format(base_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    cells_file = pl.Path("{}/cells.parquet".format(base_resolution_dir))
    cells = pd.read_parquet(cells_file)
    
    ldd_file = pl.Path("{}/ldd.parquet".format(base_resolution_dir))
    ldd = pd.read_parquet(ldd_file)
    
    # Add average discharge to cells
    discharge_file = pl.Path("{}/discharge_dailyTot_output.nc".format(save_resolution_dir))
    dataset = nc.Dataset(discharge_file)
    
    cells["x"] = None    
    lons = np.array(dataset.variables["lon"][:])
    for lon in cells["lon"].unique():
        lon_diff = np.abs(lons - lon)
        lon_index = np.where(lon_diff == np.min(lon_diff))[0][0]
        cells.loc[cells["lon"] == lon, "x"] = lon_index
        
    cells["y"] = None
    lats = np.array(dataset.variables["lat"][:])
    for lat in cells["lat"].unique():
        lat_diff = np.abs(lats - lat)
        lat_index = np.where(lat_diff == np.min(lat_diff))[0][0]
        cells.loc[cells["lat"] == lat, "y"] = lat_index
        
    discharge = dataset.variables["discharge"][:]
    if len(discharge.shape) > 2:
        discharge = discharge[0, ...]
    discharge = discharge[cells["y"].to_numpy().astype("int"),
                          cells["x"].to_numpy().astype("int")].flatten()
    cells["discharge"] = discharge
    dataset.close()
    
    routing_type, details = list(routing_details.items())[0]
    for routing_type, details in routing_details.items():
        print("\tRouting type: {}".format(routing_type))
    
        out_routing_dir = pl.Path("{}/{}".format(out_resolution_dir, routing_type))
        
        routing_number = details["routing_number"]
        base_fraction = details["base_fraction"]
        resolution_fraction = details["resolution_fractions"][resolution]
        
        ldd_sel = ldd["waterbody_type"] == routing_number
        if routing_type != "river":
            outflow_sel = np.logical_or(ldd["pit"], ldd["waterbody_transfer"])
            ldd_sel = np.logical_and(ldd_sel, outflow_sel)
        
        ldd_routing = ldd[ldd_sel]
        routing_cells = cells.loc[ldd_routing.index]
        
        # Sort cells by discharge bins
        routing_cells['bin'] = pd.cut(routing_cells['discharge'], nbins)
        
        subset_cells_list = []
        train_cells_list = []
        validate_cells_list = []
        test_cells_list = []
        
        transfer_nsamples = 0
        discharge_bin = list(reversed(routing_cells['bin'].unique()))[0]
        for discharge_bin in reversed(routing_cells['bin'].unique()):
            
            bin_cells = routing_cells.loc[routing_cells["bin"] == discharge_bin]
            bin_fraction = 1 / nbins
            
            subset_nsamples = int(routing_cells.index.size * bin_fraction * base_fraction * resolution_fraction)
            subset_nsamples += transfer_nsamples
            
            transfer_nsamples = max(0, subset_nsamples - bin_cells.index.size)
            print("Transferring {} out of {} cells to lower bin".format(transfer_nsamples, subset_nsamples))
            if transfer_nsamples > 0:
                subset_nsamples = bin_cells.index.size
            
            test_nsamples = int(subset_nsamples * test_fraction)
            train_nsamples = subset_nsamples - test_nsamples
            train_validate_nsamples = int(train_nsamples * validate_fraction)
            train_train_nsamples = train_nsamples - train_validate_nsamples
            
            subset_cells = bin_cells.sample(n = subset_nsamples,
                                            random_state=seed,
                                            axis=0)
            subset_cells_list.append(subset_cells.copy())
            
            train_cells = subset_cells.sample(n = train_train_nsamples,
                                            random_state=seed,
                                            axis=0)
            subset_cells = subset_cells.drop(train_cells.index,
                                            axis=0)
            train_cells_list.append(train_cells.copy())
            
            validate_cells = subset_cells.sample(n = train_validate_nsamples,
                                            random_state=seed,
                                            axis=0)
            subset_cells = subset_cells.drop(validate_cells.index,
                                            axis=0)
            validate_cells_list.append(validate_cells.copy())
            
            test_cells = subset_cells.sample(n = test_nsamples,
                                            random_state=seed,
                                            axis=0)
            subset_cells = subset_cells.drop(test_cells.index,
                                            axis=0)
            test_cells_list.append(test_cells.copy())
    
        subset_cells = pd.concat(subset_cells_list)
        subset_cells = subset_cells[cells.columns[:-3]]
        subset_cells_out = pl.Path("{}/cells.parquet".format(out_routing_dir))
        subset_cells_out.parent.mkdir(parents=True, exist_ok=True)
        subset_cells.to_parquet(subset_cells_out)
        
        train_cells = pd.concat(train_cells_list)
        train_cells = train_cells[cells.columns[:-3]]
        train_cells = train_cells.sort_index()
        train_cells["start"] = time_start
        train_cells["end"] = time_split
        
        validate_cells = pd.concat(validate_cells_list)
        validate_cells = validate_cells[cells.columns[:-3]]
        validate_cells = validate_cells.sort_index()
        validate_cells["start"] = time_start
        validate_cells["end"] = time_split
        
        test_cells = pd.concat(test_cells_list)
        test_cells = test_cells[cells.columns[:-3]]
        test_cells = test_cells.sort_index()
        test_cells["start"] = time_start
        test_cells["end"] = time_end
    
        print("samples: train {}, validate {} and test {}".format(train_cells.index.size,
                                                                  validate_cells.index.size,
                                                                  test_cells.index.size))
        
        train_cells_out = pl.Path("{}/train/cells.parquet".format(out_routing_dir))
        train_cells_out.parent.mkdir(parents=True, exist_ok=True)
        train_cells.to_parquet(train_cells_out)
        
        validate_cells_out = pl.Path("{}/validate/cells.parquet".format(out_routing_dir))
        validate_cells_out.parent.mkdir(parents=True, exist_ok=True)
        validate_cells.to_parquet(validate_cells_out)
        
        test_cells_out = pl.Path("{}/test/cells.parquet".format(out_routing_dir))
        test_cells_out.parent.mkdir(parents=True, exist_ok=True)
        test_cells.to_parquet(test_cells_out)
