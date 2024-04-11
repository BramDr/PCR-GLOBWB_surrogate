import pathlib as pl
import shutil
import pickle

import numpy as np

save_dir = pl.Path("./saves/train-test2")

### Input ###
template_files = [file for file in save_dir.rglob("landsurface_runoff_flux.npy") if file.is_file()]

template_file = template_files[0]
for template_file in template_files:
    
    meta_file = pl.Path("{}/{}_meta.pkl".format(template_file.parent, template_file.stem))
    
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    
    if template_file.parent.parent.stem == "input":
        area_file = pl.Path("{}/input/routing_cellAreaSum.npy".format(template_file.parent.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent.parent))
    else:
        area_file = pl.Path("{}/input/routing_cellAreaSum.npy".format(template_file.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent))
        
    area = np.load(area_file)
    with open(area_meta_file, "rb") as f:
        area_meta = pickle.load(f)
    area_sel = np.isin(area_meta["samples"], meta["samples"])
    area = area[:, area_sel]
    
    flux_features = ["landsurface_runoff",
                     "surface_water_abstraction",
                     "surface_water_infiltration",
                     "waterbody_actual_evaporation",
                     "upstream_discharge"]
    
    for feature in flux_features:
        flux_file = pl.Path("{}/{}_flux.npy".format(template_file.parent, feature))
        mean_file = pl.Path("{}/{}_mean.npy".format(template_file.parent, feature))
        
        mean_meta_file = pl.Path("{}/{}_meta.pkl".format(mean_file.parent, mean_file.stem))
        if not mean_file.exists() or not mean_meta_file.exists():
            print(mean_file)
            flux = np.load(flux_file)
            mean = flux * (86400 / area)
            np.save(mean_file, mean)
            shutil.copyfile(meta_file, mean_meta_file)


### Output ###
template_files = [file for file in save_dir.rglob("surface_water_storage_mean.npy") if file.is_file()]

template_file = template_files[0]
for template_file in template_files:
    
    meta_file = pl.Path("{}/{}_meta.pkl".format(template_file.parent, template_file.stem))
    
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    
    if template_file.parent.parent.stem == "output":
        area_file = pl.Path("{}/input/routing_cellAreaSum.npy".format(template_file.parent.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent.parent))
    else:
        area_file = pl.Path("{}/input/routing_cellAreaSum.npy".format(template_file.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent))
        
    area = np.load(area_file)
    with open(area_meta_file, "rb") as f:
        area_meta = pickle.load(f)
    area_sel = np.isin(area_meta["samples"], meta["samples"])
    area = area[:, area_sel]
    
    flux_features = ["discharge"]
    
    for feature in flux_features:
        flux_file = pl.Path("{}/{}_flux.npy".format(template_file.parent, feature))
        mean_file = pl.Path("{}/{}_mean.npy".format(template_file.parent, feature))
        
        mean_meta_file = pl.Path("{}/{}_meta.pkl".format(mean_file.parent, mean_file.stem))
        if not mean_file.exists() or not mean_meta_file.exists():
            print(mean_file)
            flux = np.load(flux_file)
            mean = flux * (86400 / area)
            np.save(mean_file, mean)
            shutil.copyfile(meta_file, mean_meta_file)
    
    mean_features = ["surface_water_storage"]
    
    for feature in mean_features:
        mean_file = pl.Path("{}/{}_mean.npy".format(template_file.parent, feature))
        flux_file = pl.Path("{}/{}_flux.npy".format(template_file.parent, feature))
        
        flux_meta_file = pl.Path("{}/{}_meta.pkl".format(flux_file.parent, flux_file.stem))
        if not flux_file.exists() or not flux_meta_file.exists():
            print(flux_file)
            mean = np.load(mean_file)
            flux = mean * (area / 86400)
            np.save(flux_file, flux)
            shutil.copyfile(meta_file, flux_meta_file)
