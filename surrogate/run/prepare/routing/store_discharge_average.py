import pathlib as pl
import shutil

import xarray as xr
import numpy as np

save_dir = pl.Path("../../../../PCR-GLOBWB/output")
out_dir = pl.Path("./saves")

resolutions = ["30min", "05min", "30sec"]

for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    save_resolution_dir = pl.Path("{}/global_{}".format(save_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))
    subset_resolution_dir = pl.Path("{}/tmp".format(out_resolution_dir))
    
    out_file = pl.Path("{}/discharge_dailyTot_output.nc".format(out_resolution_dir))
    out_tmp = pl.Path("{}/discharge_dailyTot_output.tmp.nc".format(out_resolution_dir))
    if out_file.exists():
        print("Already exists")
        continue
    
    discharge_files = [file for file in save_resolution_dir.rglob("*discharge_dailyTot_output.nc") if file.is_file()]
    discharge_files = np.sort(discharge_files)
    
    subset_files = []
    
    i = 0
    discharge_file = discharge_files[i]
    for i, discharge_file in enumerate(discharge_files):
        print("\tFile: {}".format(discharge_file))
        
        subset_file = pl.Path("{}/discharge_dailyTot_output.{}.nc".format(subset_resolution_dir, i))
        subset_tmp = pl.Path("{}/discharge_dailyTot_output.{}.tmp.nc".format(subset_resolution_dir, i))
        subset_files.append(subset_file)
        if subset_file.exists():
            print("Already exists")
            continue
        
        dataset = xr.open_dataset(discharge_file)
        dataset = dataset.mean("time")
        
        subset_file.parent.mkdir(parents = True, exist_ok = True)
        dataset.to_netcdf(subset_tmp)
        shutil.move(subset_tmp, subset_file)
    
    datasets = [xr.open_dataset(file) for file in subset_files]
    dataset = xr.merge(datasets)
    
    out_file.parent.mkdir(parents = True, exist_ok = True)
    dataset.to_netcdf(out_tmp)
    shutil.move(out_tmp, out_file)
    