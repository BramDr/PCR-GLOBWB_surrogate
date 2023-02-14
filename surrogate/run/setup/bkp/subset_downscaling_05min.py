import pathlib as pl
import subprocess
import os
from PIL import Image

import numpy as np
import pcraster as pcr

save_dir = pl.Path("./input")
tmp_dir = pl.Path("./tmp")
dir_out = pl.Path("./saves/global_05min")

clonemap_dir = pl.Path("{}/global_parallelization".format(save_dir))
mask_dir = pl.Path("{}/global_parallelization".format(save_dir))

cellSize_file = pl.Path("{}/cellsize05min.correct.map".format(save_dir))
cellSize = pcr.readmap(str(cellSize_file))

anomalyDEM_file = pl.Path("{}/anomalyDEM.map".format(save_dir))
anomalyDEM = pcr.readmap(str(anomalyDEM_file))

temperLapseRateNC_file = pl.Path("{}/temperLapseRateNC.npy".format(save_dir))
temperLapseRateNC = np.load(temperLapseRateNC_file)

temperatCorrelNC_file = pl.Path("{}/temperatCorrelNC.npy".format(save_dir))
temperatCorrelNC = np.load(temperatCorrelNC_file)

temperLapseRateNCs = []
temperatCorrelNCs = []

t = 0
for t in range(12):
    temperLapseRateNC_tmp = temperLapseRateNC[t, ...]
    temperLapseRateNC_tmp = pcr.numpy2pcr(pcr.Scalar, temperLapseRateNC_tmp, -999.9)
    temperLapseRateNCs.append(temperLapseRateNC_tmp)
    
    temperatCorrelNC_tmp = temperatCorrelNC[t, ...]
    temperatCorrelNC_tmp = pcr.numpy2pcr(pcr.Scalar, temperatCorrelNC_tmp, -999.9)
    temperatCorrelNCs.append(temperatCorrelNC_tmp)

submasks = [dir.stem for dir in dir_out.iterdir() if dir.is_dir()]
submasks = ["cells_M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))

    submask_tmp = pl.Path("{}/{}".format(tmp_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))
    
    submask_code = submask.split("_")[1]
    
    cloneMap_submask_file = pl.Path("{}/clone_{}.map".format(clonemap_dir, submask_code))
    pcr.setclone(str(cloneMap_submask_file))
    cloneMap_submask = pcr.readmap(str(cloneMap_submask_file))
    
    cloneMap_submask_out = pl.Path("{}/{}.map".format(submask_out, cloneMap_submask_file.stem))
    if cloneMap_submask_out.exists():
        os.remove(cloneMap_submask_out)
    cloneMap_submask_out.parent.mkdir(parents=True, exist_ok=True)
    pcr.report(cloneMap_submask, str(cloneMap_submask_out))
    
    maskMap_submask_file = pl.Path("{}/mask_{}.map".format(mask_dir, submask_code))
    maskMap_submask = pcr.readmap(str(maskMap_submask_file))
    
    maskMap_submask_out = pl.Path("{}/{}.map".format(submask_out, maskMap_submask_file.stem))
    if maskMap_submask_out.exists():
        os.remove(maskMap_submask_out)
    maskMap_submask_out.parent.mkdir(parents=True, exist_ok=True)
    pcr.report(maskMap_submask, str(maskMap_submask_out))
    
    cOut,err = subprocess.Popen('mapattr -p {}'.format(str(cloneMap_submask_file)), stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    cellsize = float(cOut.split()[7])
    cellsize = round(cellsize * 360000.)/360000.
    mapAttr = {'cellsize': float(cellsize),
               'rows'    : float(cOut.split()[3]),
               'cols'    : float(cOut.split()[5]),
               'xUL'     : float(cOut.split()[17]),
               'yUL'     : float(cOut.split()[19])}
    
    xmin = mapAttr['xUL']
    ymin = mapAttr['yUL'] - mapAttr['rows']*mapAttr['cellsize']
    xmax = mapAttr['xUL'] + mapAttr['cols']*mapAttr['cellsize']
    ymax = mapAttr['yUL']
    te = '-te {} {} {} {}'.format(xmin, ymin, xmax, ymax)
    
    # anomaly
    in_tmp = pl.Path("{}/{}_in.tif".format(submask_tmp, anomalyDEM_file.stem))
    if in_tmp.exists():
        os.remove(in_tmp)
    in_tmp.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = pl.Path("{}/{}_out.tif".format(submask_tmp, anomalyDEM_file.stem))
    if out_tmp.exists():
        os.remove(out_tmp)
    out_tmp.parent.mkdir(parents=True, exist_ok=True)
    submask_out_tmp = pl.Path("{}/{}.map".format(submask_out, anomalyDEM_file.stem))
    if submask_out_tmp.exists():
        os.remove(submask_out_tmp)
    submask_out_tmp.parent.mkdir(parents=True, exist_ok=True)
    
    co = 'gdal_translate -ot UInt32 {} {}'.format(anomalyDEM_file, in_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    co = 'gdalwarp {} {} {}'.format(te, in_tmp, out_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    submask = Image.open(out_tmp)
    submask = np.array(submask)
    submask = pcr.numpy2pcr(pcr.Scalar, submask, 255)
    submask = pcr.ifthen(maskMap_submask, submask)
    pcr.report(submask, str(submask_out_tmp))
    
    co = 'mapattr -c {} {}'.format(cloneMap_submask_file, submask_out_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    # area
    in_tmp = pl.Path("{}/{}_in.tif".format(submask_tmp, cellSize_file.stem))
    if in_tmp.exists():
        os.remove(in_tmp)
    in_tmp.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = pl.Path("{}/{}_out.tif".format(submask_tmp, cellSize_file.stem))
    if out_tmp.exists():
        os.remove(out_tmp)
    out_tmp.parent.mkdir(parents=True, exist_ok=True)
    submask_out_tmp = pl.Path("{}/{}.map".format(submask_out, cellSize_file.stem))
    if submask_out_tmp.exists():
        os.remove(submask_out_tmp)
    submask_out_tmp.parent.mkdir(parents=True, exist_ok=True)
    
    co = 'gdal_translate -ot Float32 {} {}'.format(cellSize_file, in_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    co = 'gdalwarp {} {} {}'.format(te, in_tmp, out_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    submask = Image.open(out_tmp)
    submask = np.array(submask)
    submask = pcr.numpy2pcr(pcr.Scalar, submask, np.nan)
    submask = pcr.ifthen(maskMap_submask, submask)
    pcr.report(submask, str(submask_out_tmp))
    
    co = 'mapattr -c {} {}'.format(cloneMap_submask_file, submask_out_tmp)
    cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
    
    submask_temperLapseRateNCs = []
    submask_temperatCorrelNCs = []
    
    temperLapseRateNC = temperLapseRateNCs[0]
    temperatCorrelNC = temperatCorrelNCs[0]
    for temperLapseRateNC, temperatCorrelNC in zip(temperLapseRateNCs, temperatCorrelNCs):
        
        # slope
        time_tmp = pl.Path("{}/{}_time.tif".format(submask_tmp, temperLapseRateNC_file.stem))
        if time_tmp.exists():
            os.remove(time_tmp)
        in_tmp = pl.Path("{}/{}_in.tif".format(submask_tmp, temperLapseRateNC_file.stem))
        if in_tmp.exists():
            os.remove(in_tmp)
        in_tmp.parent.mkdir(parents=True, exist_ok=True)
        out_tmp = pl.Path("{}/{}_out.tif".format(submask_tmp, temperLapseRateNC_file.stem))
        if out_tmp.exists():
            os.remove(out_tmp)
        out_tmp.parent.mkdir(parents=True, exist_ok=True)
        
        pcr.setclone(str(cellSize_file))
        pcr.report(temperLapseRateNC, str(time_tmp))
        
        co = 'gdal_translate -ot Float32 {} {}'.format(time_tmp, in_tmp)
        cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
        
        co = 'gdalwarp {} {} {}'.format(te, in_tmp, out_tmp)
        cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
        
        pcr.setclone(str(cloneMap_submask_file))
        
        submask = Image.open(out_tmp)
        submask = np.array(submask)
        submask = pcr.numpy2pcr(pcr.Scalar, submask, np.nan)
        submask = pcr.ifthen(maskMap_submask, submask)
        submask = pcr.pcr2numpy(submask, np.nan)
        
        submask_temperLapseRateNCs.append(submask)
        
        # correlation
        time_tmp = pl.Path("{}/{}_time.tif".format(submask_tmp, temperatCorrelNC_file.stem))
        if time_tmp.exists():
            os.remove(time_tmp)
        in_tmp = pl.Path("{}/{}_in.tif".format(submask_tmp, temperatCorrelNC_file.stem))
        if in_tmp.exists():
            os.remove(in_tmp)
        in_tmp.parent.mkdir(parents=True, exist_ok=True)
        out_tmp = pl.Path("{}/{}_out.tif".format(submask_tmp, temperatCorrelNC_file.stem))
        if out_tmp.exists():
            os.remove(out_tmp)
        out_tmp.parent.mkdir(parents=True, exist_ok=True)
        
        pcr.setclone(str(cellSize_file))
        pcr.report(temperLapseRateNC, str(time_tmp))
        
        co = 'gdal_translate -ot Float32 {} {}'.format(time_tmp, in_tmp)
        cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
        
        co = 'gdalwarp {} {} {}'.format(te, in_tmp, out_tmp)
        cOut,err = subprocess.Popen(co, stdout=subprocess.PIPE,stderr=open(os.devnull),shell=True).communicate()
        
        pcr.setclone(str(cloneMap_submask_file))
        
        submask = Image.open(out_tmp)
        submask = np.array(submask)
        submask = pcr.numpy2pcr(pcr.Scalar, submask, np.nan)
        submask = pcr.ifthen(maskMap_submask, submask)
        submask = pcr.pcr2numpy(submask, np.nan)
        
        submask_temperatCorrelNCs.append(submask)
    
    submask_temperLapseRateNC = np.stack(submask_temperLapseRateNCs, axis = 0)
    submask_temperatCorrelNC = np.stack(submask_temperatCorrelNCs, axis = 0)
    
    submask_temperLapseRateNC_out = pl.Path("{}/{}.npy".format(submask_out, temperLapseRateNC_file.stem))
    if submask_temperLapseRateNC_out.exists():
        os.remove(submask_temperLapseRateNC_out)
    submask_temperLapseRateNC_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(submask_temperLapseRateNC_out, submask_temperLapseRateNC)
    
    submask_temperatCorrelNC_out = pl.Path("{}/{}.npy".format(submask_out, temperatCorrelNC_file.stem))
    if submask_temperatCorrelNC_out.exists():
        os.remove(submask_temperatCorrelNC_out)
    submask_temperatCorrelNC_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(submask_temperatCorrelNC_out, submask_temperatCorrelNC)
