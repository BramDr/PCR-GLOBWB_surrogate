import pathlib as pl

import numpy as np
import pcraster as pcr


def get_pcr_coordinates(file: pl.Path) -> tuple[np.ndarray,
                                                np.ndarray]:
    pcr.setclone(str(file))
    pcrmap = pcr.readmap(str(file))
    pcrmap = pcr.defined(pcrmap) | ~pcr.defined(pcrmap)
    lons_map = pcr.xcoordinate(pcrmap)
    lats_map = pcr.ycoordinate(pcrmap)
    lons = pcr.pcr2numpy(lons_map, np.nan).astype(np.float32)
    lats = pcr.pcr2numpy(lats_map, np.nan).astype(np.float32)
    lons = np.median(lons, axis=0)
    lats = np.median(lats, axis=1)
    return lons, lats
