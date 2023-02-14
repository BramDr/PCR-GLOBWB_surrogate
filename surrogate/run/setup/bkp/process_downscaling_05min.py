import pathlib as pl

import pcraster as pcr
import numpy as np
import netCDF4 as nc


def regrid(array: np.ndarray,
           factor: int):
    shape_fine = np.array(array.shape)
    shape_fine[-1] *= factor
    shape_fine[-2] *= factor
    array_fine = np.zeros(shape=shape_fine,
                          dtype=array.dtype)
    
    if len(shape_fine) == 2:
        x_course = -1
        for x in range(0 , array_fine.shape[-2]):
                if x % factor == 0:
                    x_course += 1
                array_fine[x, :] = array[x_course, :].repeat(factor)
    elif len(shape_fine) == 3:
        for t in range(0, array_fine.shape[0]):
            x_course = -1
            for x in range(0, array_fine.shape[-2]):
                    if x % factor == 0:
                        x_course += 1
                    array_fine[t, x, :] = array[t, x_course, :].repeat(factor)

    return array_fine


save_dir=pl.Path("./input")
dir_out= pl.Path("./input")

downscaling_dir = pl.Path("{}/downscaling_from_30min".format(save_dir))

highResolutionDEMMap_file = pl.Path("{}/gtopo05min.map".format(downscaling_dir))
pcr.setclone(str(highResolutionDEMMap_file))

cellArea_file = pl.Path("{}/cellsize05min.correct.map".format(save_dir))
cellArea = pcr.readmap(str(cellArea_file))

#pcr.aguila(cellArea)

meteoDownscaleIds_file = pl.Path("{}/uniqueIds_30min.nc".format(downscaling_dir))
meteoDownscaleIds_dataset = nc.Dataset(meteoDownscaleIds_file)
meteoDownscaleIds_variable = meteoDownscaleIds_dataset.variables["uniqueIds_30min_map"]
meteoDownscaleIds = meteoDownscaleIds_variable[...]
meteoDownscaleIds_dataset.close()
meteoDownscaleIds = regrid(array = meteoDownscaleIds,
                           factor=6)
meteoDownscaleIds = pcr.numpy2pcr(pcr.Nominal, meteoDownscaleIds, 1e20)

#pcr.aguila(meteoDownscaleIds)

highResolutionDEM_file = pl.Path("{}/gtopo05min.nc".format(downscaling_dir))
highResolutionDEM_dataset = nc.Dataset(highResolutionDEM_file)
highResolutionDEM_variable = highResolutionDEM_dataset.variables["gtopo05min_map"]
highResolutionDEM = highResolutionDEM_variable[...]
highResolutionDEM_dataset.close()
highResolutionDEM = pcr.numpy2pcr(pcr.Scalar, highResolutionDEM, 1e20)
highResolutionDEM = pcr.cover(highResolutionDEM, 0.0)
highResolutionDEM = pcr.max(highResolutionDEM, 0.0)

#pcr.aguila(highResolutionDEM)

DEMAreaSum = pcr.areatotal(pcr.cover(highResolutionDEM * cellArea, 0.0), meteoDownscaleIds)
AreaSum = pcr.areatotal(pcr.cover(cellArea, 0.0), meteoDownscaleIds)
loweResolutionDEM = DEMAreaSum / AreaSum               
anomalyDEM = highResolutionDEM - loweResolutionDEM    # unit: meter

#pcr.aguila(anomalyDEM)

temperLapseRateNC_file = pl.Path("{}/temperature_slope.nc".format(downscaling_dir))
temperLapseRateNC_dataset = nc.Dataset(temperLapseRateNC_file)
temperLapseRateNC_variable = temperLapseRateNC_dataset.variables["temperature"]
temperLapseRateNC = temperLapseRateNC_variable[...]
temperLapseRateNC_dataset.close()
temperLapseRateNC = regrid(array = temperLapseRateNC,
                           factor=6)

temperatCorrelNC_file = pl.Path("{}/temperature_correl.nc".format(downscaling_dir))
temperatCorrelNC_dataset = nc.Dataset(temperatCorrelNC_file)
temperatCorrelNC_variable = temperatCorrelNC_dataset.variables["temperature"]
temperatCorrelNC = temperatCorrelNC_variable[...]
temperatCorrelNC_dataset.close()
temperatCorrelNC = regrid(array = temperatCorrelNC,
                           factor=6)

anomalyDEM_out = pl.Path("{}/anomalyDEM.map".format(dir_out))
anomalyDEM_out.parent.mkdir(parents=True, exist_ok=True)
pcr.report(anomalyDEM, str(anomalyDEM_out))

temperLapseRateNC_out = pl.Path("{}/temperLapseRateNC.npy".format(dir_out))
temperLapseRateNC_out.parent.mkdir(parents=True, exist_ok=True)
np.save(temperLapseRateNC_out, temperLapseRateNC)

temperatCorrelNC_out = pl.Path("{}/temperatCorrelNC.npy".format(dir_out))
temperatCorrelNC_out.parent.mkdir(parents=True, exist_ok=True)
np.save(temperatCorrelNC_out, temperatCorrelNC)