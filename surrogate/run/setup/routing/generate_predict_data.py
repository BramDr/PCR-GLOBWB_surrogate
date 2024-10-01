import pathlib as pl
import shutil
import pickle

import numpy as np

save_dir = pl.Path("./saves/predict")

### Input ###
template_files = [file for file in save_dir.rglob("landsurface_runoff_flux.npz") if file.is_file()]

template_file = template_files[0]
for template_file in template_files:
    
    meta_file = pl.Path("{}/{}_meta.pkl".format(template_file.parent, template_file.stem))
    
    with open(meta_file, "rb") as f:
        sequence_metas = pickle.load(f)
    
    if template_file.parent.parent.stem == "input":
        area_file = pl.Path("{}/input/routing_cellAreaSum.npz".format(template_file.parent.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent.parent))
    else:
        area_file = pl.Path("{}/input/routing_cellAreaSum.npz".format(template_file.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent))
    
    sequence_areas = np.load(area_file, allow_pickle=True)
    with open(area_meta_file, "rb") as f:
        sequence_area_metas = pickle.load(f)
            
    flux_features = ["landsurface_runoff",
                     "surface_water_abstraction",
                     "surface_water_infiltration",
                     "waterbody_actual_evaporation",
                     "upstream_discharge"]
    
    for feature in flux_features:
        flux_file = pl.Path("{}/{}_flux.npz".format(template_file.parent, feature))
        mean_file = pl.Path("{}/{}_mean.npz".format(template_file.parent, feature))        
        mean_meta_file = pl.Path("{}/{}_meta.pkl".format(mean_file.parent, mean_file.stem))
        
        if not mean_file.exists() or not mean_meta_file.exists():
            print(mean_file)
            sequence_fluxes = np.load(flux_file, allow_pickle=True)
            sequence_means = {}
            
            for sequence in sequence_fluxes.keys():
                flux = sequence_fluxes[sequence]
                meta = sequence_metas[sequence]
                area = sequence_areas[sequence]
                area_meta = sequence_area_metas[sequence]
                
                area_sel = np.isin(area_meta["samples"], meta["samples"])
                area = area[:, area_sel]
                mean = flux * (86400 / area)
                
                sequence_means[sequence] = mean
                    
            np.savez_compressed(mean_file, **sequence_means)
            del sequence_means
            shutil.copyfile(meta_file, mean_meta_file)


### Output ###
template_files = [file for file in save_dir.rglob("surface_water_storage_mean.npz") if file.is_file()]

template_file = template_files[0]
for template_file in template_files:
    
    meta_file = pl.Path("{}/{}_meta.pkl".format(template_file.parent, template_file.stem))
    
    with open(meta_file, "rb") as f:
        sequence_metas = pickle.load(f)
    
    if template_file.parent.parent.stem == "output":
        area_file = pl.Path("{}/input/routing_cellAreaSum.npz".format(template_file.parent.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent.parent))
    else:
        area_file = pl.Path("{}/input/routing_cellAreaSum.npz".format(template_file.parent.parent))
        area_meta_file = pl.Path("{}/input/routing_cellAreaSum_meta.pkl".format(template_file.parent.parent))
        
    sequence_areas = np.load(area_file, allow_pickle=True)
    with open(area_meta_file, "rb") as f:
        sequence_area_metas = pickle.load(f)
    
    flux_features = ["discharge"]
    
    for feature in flux_features:
        flux_file = pl.Path("{}/{}_flux.npz".format(template_file.parent, feature))
        mean_file = pl.Path("{}/{}_mean.npz".format(template_file.parent, feature))
        mean_meta_file = pl.Path("{}/{}_meta.pkl".format(mean_file.parent, mean_file.stem))
        
        if not mean_file.exists() or not mean_meta_file.exists():
            print(mean_file)
            sequence_fluxes = np.load(flux_file, allow_pickle=True)
            sequence_means = {}
            
            for sequence in sequence_fluxes.keys():
                flux = sequence_fluxes[sequence]
                meta = sequence_metas[sequence]
                area = sequence_areas[sequence]
                area_meta = sequence_area_metas[sequence]
                
                area_sel = np.isin(area_meta["samples"], meta["samples"])
                area = area[:, area_sel]
                mean = flux * (86400 / area)
                
                sequence_means[sequence] = mean
                    
            np.savez_compressed(mean_file, **sequence_means)
            del sequence_means
            shutil.copyfile(meta_file, mean_meta_file)
    
    mean_features = ["surface_water_storage"]
    
    for feature in mean_features:
        flux_file = pl.Path("{}/{}_flux.npz".format(template_file.parent, feature))
        mean_file = pl.Path("{}/{}_mean.npz".format(template_file.parent, feature))        
        flux_meta_file = pl.Path("{}/{}_meta.pkl".format(flux_file.parent, flux_file.stem))
        
        if not flux_file.exists() or not flux_meta_file.exists():
            print(flux_file)
            sequence_means = np.load(mean_file, allow_pickle=True)
            sequence_fluxes = {}
            
            for sequence in sequence_means.keys():
                mean = sequence_means[sequence]
                meta = sequence_metas[sequence]
                area = sequence_areas[sequence]
                area_meta = sequence_area_metas[sequence]
                
                area_sel = np.isin(area_meta["samples"], meta["samples"])
                area = area[:, area_sel]
                flux = mean * (area / 86400)
                
                sequence_fluxes[sequence] = flux
                    
            np.savez_compressed(flux_file, **sequence_fluxes)
            del sequence_fluxes
            shutil.copyfile(meta_file, flux_meta_file)
        