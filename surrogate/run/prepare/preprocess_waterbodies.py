import pathlib as pl

import numpy as np
import pcraster as pcr
import netCDF4 as nc

input_dir = pl.Path("./input")
out_dir = pl.Path("./saves")

resolutions = [dir.stem for dir in input_dir.iterdir() if dir.is_dir()]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    input_resolution_dir = pl.Path("{}/{}".format(input_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))
    
    ldd_file = pl.Path("{}/lddsound.map".format(input_resolution_dir))
    pcr.setclone(str(ldd_file))
    ldd = pcr.readmap(str(ldd_file))
    
    area_file = pl.Path("{}/cellArea.nc".format(input_resolution_dir))
    with nc.Dataset(area_file) as area_dataset:
        if resolution == "30min":
            area_variable = area_dataset.variables["cellarea30min_map"]
        elif resolution == "05min":
            area_variable = area_dataset.variables["cellsize05min_correct_map"]
        elif resolution == "30sec":
            area_variable = area_dataset.variables["cell_area"]
        else:
            raise ValueError("Unknown resolution: {}".format(resolution))
        area = area_variable[:]
        area[area.mask] = 0
        area = np.array(area, dtype=np.float32)
    
    waterbody_file = pl.Path("{}/waterBodies.nc".format(input_resolution_dir))
    with nc.Dataset(waterbody_file) as waterbody_dataset:
        waterbody_variable = waterbody_dataset.variables["waterBodyTyp"]
        waterbody_type = waterbody_variable[:][0, ...]
        waterbody_type[waterbody_type[:].mask] = -1
        waterbody_type = np.array(waterbody_type, dtype=np.int32)
        waterbody_variable = waterbody_dataset.variables["waterBodyIds"]
        waterbody_id = waterbody_variable[:][0, ...]
        waterbody_id[waterbody_id[:].mask] = -1
        waterbody_id = np.array(waterbody_id, dtype=np.int32)
        waterbody_variable = waterbody_dataset.variables["fracWaterInp"]
        waterbody_fraction = waterbody_variable[:][0, ...]
        waterbody_fraction[waterbody_fraction[:].mask] = -1
        waterbody_fraction = np.array(waterbody_fraction, dtype=np.float32)
        waterbody_variable = waterbody_dataset.variables["resSfAreaInp"]
        waterbody_resarea = waterbody_variable[:][0, ...]
        waterbody_resarea[waterbody_resarea[:].mask] = -1
        waterbody_resarea = np.array(waterbody_resarea, dtype=np.float32)
        waterbody_variable = waterbody_dataset.variables["resMaxCapInp"]
        waterbody_rescap = waterbody_variable[:][0, ...]
        waterbody_rescap[waterbody_rescap[:].mask] = -1
        waterbody_rescap = np.array(waterbody_rescap, dtype=np.float32)
    
    area = pcr.numpy2pcr(dataType=pcr.Scalar, array=area, mv=0)
    waterbody_type = pcr.numpy2pcr(dataType=pcr.Nominal, array=waterbody_type, mv=-1)
    waterbody_id = pcr.numpy2pcr(dataType=pcr.Nominal, array=waterbody_id, mv=-1)
    waterbody_fraction = pcr.numpy2pcr(dataType=pcr.Scalar, array=waterbody_fraction, mv=-1)
    waterbody_resarea = pcr.numpy2pcr(dataType=pcr.Scalar, array=waterbody_resarea, mv=-1)
    waterbody_rescap = pcr.numpy2pcr(dataType=pcr.Scalar, array=waterbody_rescap, mv=-1)
    waterbody_id_orig = waterbody_id
    
    # pcr.aguila(area)
    # pcr.aguila(waterbody_type)
    # pcr.aguila(waterbody_id)
    # pcr.aguila(waterbody_fraction)
    
    # Find waterbody outflow points
    wbCatchment = pcr.catchmenttotal(pcr.scalar(1), ldd)
    waterbody_out = pcr.ifthen(wbCatchment == pcr.areamaximum(wbCatchment,
                                                              waterbody_id),
                               waterbody_id)
    waterbody_out = pcr.ifthen(pcr.areaorder(pcr.scalar(waterbody_out),
                                            waterbody_out) == 1.,
                               waterbody_out)
    waterbody_out = pcr.ifthen(pcr.scalar(waterbody_id) > 0.,
                               waterbody_out)
    
    # Correct waterbodies based on outflow points
    waterbody_id = pcr.ifthen(pcr.scalar(waterbody_id) > 0.,
                              pcr.subcatchment(ldd,waterbody_out))
    waterbody_out = pcr.ifthen(pcr.scalar(waterbody_out) > 0.,
                               pcr.spatial(pcr.boolean(1)))
    waterbody_id_out = waterbody_id
    waterbody_id_diff = pcr.scalar(waterbody_id_orig) - pcr.scalar(waterbody_id_out) != 0
    # pcr.aguila(waterbody_id_diff)
    print(pcr.pcr2numpy(waterbody_id_diff, mv = np.nan).sum())
    
    # Calculate waterbody area
    waterbody_resarea = pcr.areaaverage(waterbody_resarea,
                                        waterbody_id)                        
    waterbody_resarea = pcr.cover(waterbody_resarea,0.)
    waterbody_area = pcr.max(pcr.areatotal(pcr.cover(waterbody_fraction * area, 0.0),
                                           waterbody_id),
                             pcr.areaaverage(pcr.cover(waterbody_resarea, 0.0),
                                             waterbody_id))
    waterbody_area = pcr.ifthen(waterbody_area > 0.,\
                                waterbody_area)
                            
    # Correct waterbodies based on area
    waterbody_id = pcr.ifthen(waterbody_area > 0.,
                              waterbody_id)               
    waterbody_out = pcr.ifthen(pcr.boolean(waterbody_id),
                               waterbody_out)
    waterbody_id_area = waterbody_id
    waterbody_id_diff = pcr.scalar(waterbody_id_out) - pcr.scalar(waterbody_id_area) != 0
    # pcr.aguila(waterbody_id_diff)
    print(pcr.pcr2numpy(waterbody_id_diff, mv = np.nan).sum())
    
    # Correct waterbody types based on type majority
    waterbody_type = pcr.ifthen(pcr.scalar(waterbody_type) > 0,
                                pcr.nominal(waterbody_type))    
    waterbody_type = pcr.ifthen(pcr.scalar(waterbody_id) > 0,
                                pcr.nominal(waterbody_type))    
    waterbody_type = pcr.areamajority(waterbody_type,
                                      waterbody_id)
    waterbody_type = pcr.ifthen(pcr.scalar(waterbody_type) > 0,
                                pcr.nominal(waterbody_type))    
    waterbody_type = pcr.ifthen(pcr.boolean(waterbody_id),
                                waterbody_type)

    # Correct waterbodies based on type
    waterbody_id = pcr.ifthen(pcr.scalar(waterbody_type) > 0,
                              waterbody_id)               
    waterbody_out = pcr.ifthen(pcr.scalar(waterbody_id) > 0,
                               waterbody_out)
    waterbody_id_type = waterbody_id
    waterbody_id_diff = pcr.scalar(waterbody_id_area) - pcr.scalar(waterbody_id_type) != 0
    # pcr.aguila(waterbody_id_diff)
    print(pcr.pcr2numpy(waterbody_id_diff, mv = np.nan).sum())
    
    # Calculate waterbody capacity
    waterbody_rescap = pcr.ifthen(waterbody_rescap > 0.,
                                  waterbody_rescap)
    waterbody_rescap = pcr.areaaverage(waterbody_rescap,
                                       waterbody_id)
                                        
    waterbody_cap = pcr.cover(waterbody_rescap, 0.0)
    waterbody_cap = pcr.ifthen(pcr.boolean(waterbody_id),
                                           waterbody_cap)

    waterbody_type = pcr.ifthen(pcr.scalar(waterbody_type) > 0., 
                                waterbody_type) 
    waterbody_type = pcr.ifthenelse(waterbody_cap > 0.,
                                    waterbody_type,
                                    pcr.ifthenelse(pcr.scalar(waterbody_type) == 2,
                                                   pcr.nominal(1),
                                                   waterbody_type)) 

    # Correct waterbodies based on capacity
    waterbody_type = pcr.ifthen(waterbody_area > 0.,
                                waterbody_type)
    waterbody_type = pcr.ifthen(pcr.scalar(waterbody_type) > 0.,
                                waterbody_type)
    waterbody_id = pcr.ifthen(pcr.scalar(waterbody_type) > 0.,
                              waterbody_id)
    waterbody_out = pcr.ifthen(pcr.scalar(waterbody_id) > 0.,
                               waterbody_out)
    waterbody_id_cap = waterbody_id
    waterbody_id_diff = pcr.scalar(waterbody_id_type) - pcr.scalar(waterbody_id_cap) != 0
    # pcr.aguila(waterbody_id_diff)
    print(pcr.pcr2numpy(waterbody_id_diff, mv = np.nan).sum())
    
    out_resolution_dir.mkdir(parents=True, exist_ok=True)
    area_out = pl.Path("{}/cell_area.map".format(out_resolution_dir))
    pcr.report(area, str(area_out))
    waterbody_type_out = pl.Path("{}/waterbody_type.map".format(out_resolution_dir))
    pcr.report(waterbody_type, str(waterbody_type_out))
    waterbody_id_out = pl.Path("{}/waterbody_id.map".format(out_resolution_dir))
    pcr.report(waterbody_id, str(waterbody_id_out))
    waterbody_out_out = pl.Path("{}/waterbody_out.map".format(out_resolution_dir))
    pcr.report(waterbody_out, str(waterbody_out_out))