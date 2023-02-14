import pathlib as pl
import pickle
import shutil

import pcraster as pcr
import pandas as pd
import numpy as np

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
maxCorrelationCriteria = -0.75

submasks = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
submasks = ["cells_M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir=pl.Path("{}/{}".format(save_dir, submask))
    feature_submask_dir = pl.Path("{}/{}".format(feature_dir,submask))
    
    input_dir=pl.Path("{}/input".format(submask_dir))
    
    cells_file = pl.Path("{}/cells.csv".format(feature_submask_dir))
    cells = pd.read_csv(cells_file, index_col=0)
    
    # Backup temperature
    temperature_file = pl.Path("{}/meteoOptions/temperaturenc/temperature.npy".format(input_dir))
    temperature_meta_file = pl.Path("{}/meteoOptions/temperaturenc/temperature_meta.pkl".format(input_dir))
    temperature_bkp = pl.Path("{}/meteoOptions/temperaturenc/temperature.bnpy".format(input_dir))
    temperature_meta_bkp = pl.Path("{}/meteoOptions/temperaturenc/temperature_meta.bpkl".format(input_dir))
    if not temperature_bkp.exists():
        shutil.copy(temperature_file, temperature_bkp)
    if not temperature_meta_bkp.exists():
        shutil.copy(temperature_meta_file, temperature_meta_bkp)
    
    temperature = np.load(temperature_bkp)
    with open(temperature_meta_bkp, "rb") as file:
        temperature_meta = pickle.load(file)
    
    anomalyDEM_file = pl.Path("{}/anomalyDEM.map".format(submask_dir))
    anomalyDEM = pcr.readmap(str(anomalyDEM_file))
    anomalyDEM = pcr.pcr2numpy(anomalyDEM, np.nan)
    
    tmpSlope_file = pl.Path("{}/temperLapseRateNC.npy".format(submask_dir))
    tmpSlope = np.load(tmpSlope_file)
    tmpSlope = np.where(tmpSlope > 0, 0, tmpSlope)
    
    tmpCriteria_file = pl.Path("{}/temperatCorrelNC.npy".format(submask_dir))
    tmpCriteria = np.load(tmpCriteria_file)
    tmpSlope = np.where(tmpCriteria < maxCorrelationCriteria, tmpSlope, 0)
    tmpSlope = np.where(np.isnan(tmpSlope), 0, tmpSlope)
    

    temperature_self = temperature[temperature_meta["spatial_mapping"], :]
    anomalyDEM_self = anomalyDEM[cells["y"], cells["x"]]
    anomalyDEM_self = np.expand_dims(anomalyDEM_self, axis = -1)
    tmpSlope_self = tmpSlope[:, cells["y"], cells["x"]]
    month_indices = [datum.month - 1 for datum in temperature_meta["dates"]]
    tmpSlope_self = tmpSlope_self[month_indices, :]
    tmpSlope_self = np.transpose(a = tmpSlope_self, axes = [1, 0])
    temperature_self = temperature_self + tmpSlope_self * anomalyDEM_self
    
    dem_file = pl.Path("{}/meteoDownscalingOptions/highresolutiondem/gtopo05min.npy".format(submask_dir))
    dem = np.load(dem_file)    
    area_file = pl.Path("{}/routingOptions/cellAreamap/cellsize05min.correct.npy".format(submask_dir))
    area = np.load(area_file)    
    slope_file = pl.Path("{}/meteoDownscalingOptions/temperlapseratenc/temperature.npy".format(submask_dir))
    slope = np.load(slope_file)    
    corr_file = pl.Path("{}/meteoDownscalingOptions/temperatcorrelnc/temperature.npy".format(submask_dir))
    corr = np.load(corr_file)
        
    low_meta_file = pl.Path("{}/{}_meta.pkl".format(slope_file.parent, slope_file.stem))
    with open(low_meta_file, "rb") as file:
        low_meta = pickle.load(file)
        
    high_meta_file = pl.Path("{}/{}_meta.pkl".format(dem_file.parent, dem_file.stem))
    with open(high_meta_file, "rb") as file:
        high_meta = pickle.load(file)
    
    # Calculate dem anomaly
    low_dem = np.full(shape = slope.shape[:1], fill_value=0, dtype=dem.dtype)
    low_area = np.full(shape = slope.shape[:1], fill_value=0, dtype=area.dtype)
    for high_index, low_index in enumerate(low_meta["spatial_mapping"]):
        low_dem[low_index] += dem[high_index] * area[high_index]
        low_area[low_index] += area[high_index]
    low_dem /= low_area
    
    anomaly = np.full(shape = dem.shape, fill_value=np.nan, dtype=dem.dtype)
    for high_index, low_index in enumerate(low_meta["spatial_mapping"]):
        anomaly[high_index] = dem[high_index] - low_dem[low_index]
    
    # Correct slope
    slope = np.where(slope > 0, 0, slope)
    slope = np.where(corr < maxCorrelationCriteria, slope, 0)
    
    # Calculate temperature     
    months = np.array([datum.month - 1 for datum in temperature_meta["origional_dates"]])
    indices = low_meta["spatial_mapping"]
    
    slope_corr =  slope[:, months]
    anomaly_corr = np.expand_dims(a=anomaly, axis=-1)
    temperature_corr = temperature[indices, :] + slope_corr[indices, :] * anomaly_corr
    
    temperature_meta_corr = temperature_meta
    temperature_meta_corr["spatial_mapping"] = high_meta["spatial_mapping"]
    temperature_meta_corr["origional_lons"] = high_meta["origional_lons"]
    temperature_meta_corr["origional_lats"] = high_meta["origional_lats"]
    temperature_meta_corr["x_resolution"] = high_meta["x_resolution"]
    temperature_meta_corr["y_resolution"] = high_meta["y_resolution"]
    
    np.save(file=temperature_file, arr = temperature_corr)
    with open(temperature_meta_file, "wb") as file:
        pickle.dump(temperature_meta_corr, file)
    