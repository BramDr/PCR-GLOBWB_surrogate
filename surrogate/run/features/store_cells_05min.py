import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import pcraster as pcr

from utils.calculate_coordinate_indices import calculate_coordinate_indices

landmask_dir = pl.Path("../../../PCR-GLOBWB/cloneMaps/global_parallelization")
sim_time_start = dt.date(2000, 1, 1)
sim_time_end = dt.date(2010, 12, 31)
dir_out = pl.Path("./saves/global_05min")
seed = 19920223
resolution_arcseconds = 5 * 60

global_lons = np.arange(start = resolution_arcseconds / 2,
                        stop = 360 * 60 * 60,
                        step = resolution_arcseconds)
global_lats = np.arange(start = resolution_arcseconds / 2,
                        stop = 180 * 60 * 60,
                        step = resolution_arcseconds)
global_lons /= 60 * 60
global_lats /= 60 * 60
global_lons -= 180
global_lats -= 90
global_lats = np.flip(global_lats)

landmask_files = [file for file in landmask_dir.rglob("mask_*")]
landmask_files = [file for file in landmask_dir.rglob("mask_M17*")]

landmask_file = landmask_files[0]
for landmask_file in landmask_files:
    submask = landmask_file.stem.split("_")[1]

    print("Working on {}".format(submask))

    pcr.setclone(str(landmask_file))
    mask_map = pcr.readmap(str(landmask_file))
    
    lons_map = pcr.xcoordinate(mask_map)
    lats_map = pcr.ycoordinate(mask_map)
    mask = pcr.pcr2numpy(mask_map, np.nan)
    lons = pcr.pcr2numpy(lons_map, np.nan)
    lats = pcr.pcr2numpy(lats_map, np.nan)
    lat_size = lats.shape[0]
    lon_size = lons.shape[1]
    xs = np.array([x for _ in range(lat_size) for x in range(lon_size)])
    ys = np.array([y for y in range(lat_size) for _ in range(lon_size)])
    mask = mask.flatten()
    lons = lons.flatten()
    lats = lats.flatten()
    
    mask_sel = mask > 0
    mask_lons = lons[mask_sel]
    mask_lats = lats[mask_sel]
    mask_local_xs = xs[mask_sel]
    mask_local_ys = ys[mask_sel]
    mask_global_xs = calculate_coordinate_indices(mask_lons, global_lons)
    mask_global_ys = calculate_coordinate_indices(mask_lats, global_lats)

    cells_dict = {"local_x": mask_local_xs,
                  "local_y": mask_local_ys,
                  "global_x": mask_global_xs,
                  "global_y": mask_global_ys,
                  "lon": mask_lons,
                  "lat": mask_lats}
    cells = pd.DataFrame(cells_dict)
    
    cells["submask"] = submask
    cells["start"] = sim_time_start
    cells["end"] = sim_time_end

    cells_out = pl.Path("{}/{}/cells.csv".format(dir_out, submask))
    cells_out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(cells_out)
