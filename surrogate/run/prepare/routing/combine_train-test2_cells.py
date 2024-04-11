import pathlib as pl

import pandas as pd
import numpy as np
import netCDF4 as nc

base_dir = pl.Path("./saves")
save_dir = pl.Path("./saves/train-test2")
out_dir = pl.Path("./saves/train-test2/mulres")
seed = 19920223

nbins = 5

resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]
save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolutions[0]))
routing_types = [dir.stem for dir in save_resolution_dir.iterdir() if dir.is_dir()]
save_routing_dir = pl.Path("{}/{}".format(save_resolution_dir, routing_types[0]))
trainsets = [dir.stem for dir in save_routing_dir.iterdir() if dir.is_dir()]

cells_trainsets = {}

routing_type = routing_types[0]
for routing_type in routing_types:
    print("Routing type: {}".format(routing_type))
        
    out_routing_dir = pl.Path("{}/{}".format(out_dir, routing_type))
    
    trainset = trainsets[0]
    for trainset in trainsets:
        print("\tTrainset: {}".format(trainset))
        
        out_trainset_dir = pl.Path("{}/{}".format(out_routing_dir, trainset))
        
        resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]

        cells_list = []
        
        resolution = resolutions[0]
        for resolution in resolutions:
            print("\t\tResolution: {}".format(resolution))

            base_resolution_dir = pl.Path("{}/{}".format(base_dir, resolution))
            save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
            save_routing_dir = pl.Path("{}/{}".format(save_resolution_dir, routing_type))
            save_trainset_dir = pl.Path("{}/{}".format(save_routing_dir, trainset))
        
            cells_file = pl.Path("{}/cells.parquet".format(save_trainset_dir))
            cells = pd.read_parquet(cells_file)
            
            discharge_file = pl.Path("{}/discharge_dailyTot_output.nc".format(base_resolution_dir))
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
            
            # Sort cells by discharge bins
            cells['bin'] = pd.cut(cells['discharge'], nbins)
            
            transfer_nsamples = 0
            discharge_bin = list(reversed(cells['bin'].unique()))[0]
            for discharge_bin in reversed(cells['bin'].unique()):
                
                bin_cells = cells.loc[cells["bin"] == discharge_bin]
                bin_fraction = 1 / nbins
                resolution_fraction = 1 / len(resolutions)
                
                subset_nsamples = int(cells.index.size * bin_fraction * resolution_fraction)
                subset_nsamples += transfer_nsamples
                
                transfer_nsamples = max(0, subset_nsamples - bin_cells.index.size)
                print("Transferring {} out of {} cells to lower bin".format(transfer_nsamples, subset_nsamples))
                if transfer_nsamples > 0:
                    subset_nsamples = bin_cells.index.size
                
                bin_cells = bin_cells.sample(n = subset_nsamples,
                                             random_state=seed,
                                             axis = 0)
                bin_cells = bin_cells.sort_index()
                bin_cells["resolution"] = resolution
                
                cells_list.append(bin_cells)
        
        cells = pd.concat(cells_list, axis = 0)
        cells = cells.reset_index()
        cells = cells.rename({"index": "resolution_index"},
                            axis = 1)
        cells = cells.drop(["x", "y", "discharge", "bin"], axis = 1)
        
        print("samples {}".format(cells.index.size))
        
        cells_out = pl.Path("{}/cells.parquet".format(out_trainset_dir))
        cells_out.parent.mkdir(parents=True, exist_ok=True)
        cells.to_parquet(cells_out)
        