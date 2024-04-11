import pathlib as pl

import numpy as np
import pandas as pd
import pcraster as pcr

from utils.get_map_attributes import get_map_attributes

mask_dir = pl.Path("./input/30sec/europe_parallelization")
region_mask_file = pl.Path("./input/30sec/clone_europe_30sec.map")
dir_out = pl.Path("./saves/30sec")
seed = 19920223

resolution_arcseconds = 0.5 * 60
global_lons = np.arange(start=resolution_arcseconds / 2,
                        stop=360 * 60 * 60,
                        step=resolution_arcseconds)
global_lats = np.arange(start=resolution_arcseconds / 2,
                        stop=180 * 60 * 60,
                        step=resolution_arcseconds)

global_lons /= 60 * 60
global_lats /= 60 * 60
global_lons -= 180
global_lats -= 90
global_lats = np.flip(global_lats)

global_xs = np.arange(global_lons.size)
global_ys = np.arange(global_lats.size)

pcr.setclone(str(region_mask_file))
region_mask_attributes = get_map_attributes(region_mask_file)

lon_start = region_mask_attributes["x_corner_value"]
lon_start += region_mask_attributes["resolution"] / 2
lon_end = region_mask_attributes["x_corner_value"] + region_mask_attributes["resolution"] * region_mask_attributes["x_len"]
lon_end -= region_mask_attributes["resolution"] / 2
region_lons = np.linspace(lon_start,
                          lon_end,
                          region_mask_attributes["x_len"])

lat_start = region_mask_attributes["y_corner_value"]
lat_start -= region_mask_attributes["resolution"] / 2
lat_end = region_mask_attributes["y_corner_value"] - region_mask_attributes["resolution"] * region_mask_attributes["y_len"]
lat_end += region_mask_attributes["resolution"] / 2
region_lats = np.linspace(lat_start,
                          lat_end,
                          region_mask_attributes["y_len"])

region_xs = np.arange(global_lons.size)
region_ys = np.arange(global_lats.size)

cells = pd.DataFrame()

mask_files = [file for file in mask_dir.rglob("mask_*")]

mask_file = mask_files[0]
for mask_file in mask_files:
    submask = mask_file.stem.split("_")[1]

    print("Working on {}".format(submask))

    pcr.setclone(str(mask_file))
    mask_attributes = get_map_attributes(mask_file)

    mask_map = pcr.readmap(str(mask_file))
    mask = pcr.pcr2numpy(mask_map, np.nan).flatten()

    x_start = np.where(global_lons > mask_attributes["x_corner_value"])[0][0]
    x_end = x_start + mask_attributes["x_len"]
    y_start = np.where(global_lats < mask_attributes["y_corner_value"])[0][0]
    y_end = y_start + mask_attributes["y_len"]

    mask_global_lons = global_lons[x_start:x_end]
    mask_global_lats = global_lats[y_start:y_end]
    mask_global_xs = global_xs[x_start:x_end]
    mask_global_ys = global_ys[y_start:y_end]
    mask_local_xs = mask_global_xs - min(mask_global_xs)
    mask_local_ys = mask_global_ys - min(mask_global_ys)
    
    x_start = np.where(region_lons > mask_attributes["x_corner_value"])[0][0]
    x_end = x_start + mask_attributes["x_len"]
    y_start = np.where(region_lats < mask_attributes["y_corner_value"])[0][0]
    y_end = y_start + mask_attributes["y_len"]
    
    mask_region_xs = region_xs[x_start:x_end]
    mask_region_ys = region_ys[y_start:y_end]

    mask_sel = mask > 0
    mask_lons_sel = np.array(
        [lon for lat in mask_global_lats for lon in mask_global_lons])[mask_sel]
    mask_lats_sel = np.array(
        [lat for lat in mask_global_lats for lon in mask_global_lons])[mask_sel]
    mask_global_xs_sel = np.array(
        [x for y in mask_global_ys for x in mask_global_xs])[mask_sel]
    mask_global_ys_sel = np.array(
        [y for y in mask_global_ys for x in mask_global_xs])[mask_sel]
    mask_local_xs_sel = np.array(
        [x for y in mask_local_ys for x in mask_local_xs])[mask_sel]
    mask_local_ys_sel = np.array(
        [y for y in mask_local_ys for x in mask_local_xs])[mask_sel]
    mask_region_xs_sel = np.array(
        [x for y in mask_region_ys for x in mask_region_xs])[mask_sel]
    mask_region_ys_sel = np.array(
        [y for y in mask_region_ys for x in mask_region_xs])[mask_sel]

    submask_cells_dict = {"local_x": mask_local_xs_sel,
                          "local_y": mask_local_ys_sel,
                          "global_x": mask_global_xs_sel,
                          "global_y": mask_global_ys_sel,
                          "region_x": mask_region_xs_sel,
                          "region_y": mask_region_ys_sel,
                          "lon": mask_lons_sel,
                          "lat": mask_lats_sel}
    submask_cells = pd.DataFrame(submask_cells_dict)
    submask_cells["domain"] = submask
    cells = pd.concat((cells, submask_cells))

cells = cells.reset_index(drop=True)
cells["region"] = "europe_30sec"

cells = cells.astype({"local_x": "int32",
                      "local_y": "int32",
                      "global_x": "int32",
                      "global_y": "int32",
                      "region_x": "int32",
                      "region_y": "int32",
                      "lon": "float32",
                      "lat": "float32",
                      "domain": "category",
                      "region": "category"})

cells_out = pl.Path("{}/cells.parquet".format(dir_out))
cells_out.parent.mkdir(parents=True, exist_ok=True)
cells.to_parquet(cells_out)
