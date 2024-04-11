import pathlib as pl
import re

import pandas as pd
import numpy as np
import pcraster as pcr
import netCDF4 as nc

save_dir = pl.Path("./saves")
input_dir = pl.Path("./input")
out_dir = pl.Path("./saves")

resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() and dir.stem != "predict"]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
    input_resolution_dir = pl.Path("{}/{}".format(input_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    cells_file = pl.Path("{}/cells.parquet".format(save_resolution_dir))
    cells = pd.read_parquet(cells_file)
    
    ldd_file = pl.Path("{}/lddsound.map".format(input_resolution_dir))
    pcr.setclone(str(ldd_file))
    ldd = pcr.readmap(str(ldd_file))
    ldd_array = pcr.pcr2numpy(map=ldd, mv=0)
    
    mask_array = np.zeros(shape = ldd_array.shape)
    mask_array[cells["global_y"], cells["global_x"]] = True
    mask = pcr.numpy2pcr(dataType=pcr.Boolean, array=mask_array, mv=False)
    # pcr.aguila(mask)
    
    ldd = pcr.ifthen(mask, ldd)
    ldd = pcr.lddrepair(ldd)
    ldd_array = pcr.pcr2numpy(map=ldd, mv=0)

    id_array = np.full(shape=ldd_array.shape, fill_value=-1, dtype=np.int64)
    id_array[cells["global_y"], cells["global_x"]] = cells.index
    id = pcr.numpy2pcr(dataType=pcr.Scalar, array=id_array, mv=-1)
    # pcr.aguila(id)

    downstream_id = pcr.downstream(ldd, id)
    downstream_id_array = pcr.pcr2numpy(map=downstream_id, mv=-1)
    downstream_id_array = downstream_id_array.astype(np.int32)
    
    if (downstream_id_array[cells["global_y"], cells["global_x"]] < 0).sum() > 0:
        raise ValueError("Missing downstream")
    
    waterbody_type_file = pl.Path("{}/waterbody_type.map".format(save_resolution_dir))
    waterbody_id_file = pl.Path("{}/waterbody_id.map".format(save_resolution_dir))
    waterbody_out_file = pl.Path("{}/waterbody_out.map".format(save_resolution_dir))
    waterbody_type = pcr.readmap(str(waterbody_type_file))
    waterbody_id = pcr.readmap(str(waterbody_id_file))
    waterbody_out = pcr.readmap(str(waterbody_out_file))
    waterbody_type_array = pcr.pcr2numpy(map=waterbody_type, mv=0)
    waterbody_id_array = pcr.pcr2numpy(map=waterbody_id, mv=0)
    waterbody_out_array = pcr.pcr2numpy(map=waterbody_out, mv=0)
    
    ldd_df = {"waterbody_type": waterbody_type_array[cells["global_y"], cells["global_x"]],
              "waterbody_id": waterbody_id_array[cells["global_y"], cells["global_x"]],
              "waterbody_out": waterbody_out_array[cells["global_y"], cells["global_x"]],
              "downstream": downstream_id_array[cells["global_y"], cells["global_x"]]}
    ldd_df = pd.DataFrame(ldd_df)
    ldd_df.index = cells.index
    
    ldd_df["downstream_waterbody_id"] = ldd_df.loc[ldd_df["downstream"], "waterbody_id"].to_numpy()
    ldd_df["pit"] = ldd_df.index == ldd_df["downstream"]
    ldd_df["waterbody_transfer"] = ldd_df["waterbody_id"] != ldd_df["downstream_waterbody_id"]
    ldd_df["waterbody_outflow"] = np.logical_and(ldd_df["waterbody_id"] != 0,
                                                 np.logical_or(ldd_df["pit"],
                                                               ldd_df["waterbody_transfer"]))
    
    # Save
    ldd_df = ldd_df.astype({"waterbody_id": "int32",
                            "waterbody_type": "int8",
                            "waterbody_out": "bool",
                            "downstream": "int32",
                            "downstream_waterbody_id": "int32",
                            "pit": "bool",
                            "waterbody_transfer": "bool",
                            "waterbody_outflow": "bool"})
    
    ldd_out = pl.Path("{}/ldd.parquet".format(out_resolution_dir))
    ldd_out.parent.mkdir(parents=True,
                        exist_ok=True)
    ldd_df.to_parquet(ldd_out)
