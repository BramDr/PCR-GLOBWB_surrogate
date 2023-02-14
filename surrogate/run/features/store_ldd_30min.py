import pathlib as pl

import pandas as pd
import numpy as np
import pcraster as pcr

save_dir = pl.Path("./saves/global_30min")
dir_out = pl.Path("./saves/global_30min")

cells_file = pl.Path("{}/cells.csv".format(save_dir))
cells = pd.read_csv(cells_file, index_col=0)

cellSizeInArcDeg = 30 / 60
cellSizeInArcMin = cellSizeInArcDeg * 60
verticalSizeInMeter =  cellSizeInArcMin * 1852.

maskMap_file = pl.Path("{}/maskMap.map".format(save_dir))
maskMap = pcr.readmap(str(maskMap_file))

cellArea_file = pl.Path("{}/cellSize.map".format(save_dir))
cellArea = pcr.readmap(str(cellArea_file))
cellArea = pcr.ifthen(maskMap, cellArea)

lddMap_file = pl.Path("{}/lddMap.map".format(save_dir))
lddMap = pcr.readmap(str(lddMap_file))
lddMap = pcr.ifthen(maskMap, lddMap)

cellLengthFD  = ((cellArea / verticalSizeInMeter)**(2) + (verticalSizeInMeter)**(2))**(0.5) 
channelLength = cellLengthFD

lddArray = pcr.pcr2numpy(map = lddMap,
                         mv = 255)
distanceDownstreamMap = pcr.ldddist(lddMap,
                                 pcr.nominal(lddMap) == 5,
                                 channelLength)
distanceDownstreamArray = pcr.pcr2numpy(map = distanceDownstreamMap,
                                        mv = -1)

idArray = np.full(shape = distanceDownstreamArray.shape, fill_value=-1, dtype=np.int64)
idArray[cells["y"], cells["x"]] = cells.index
idMap = pcr.numpy2pcr(dataType = pcr.Scalar, array = idArray, mv = -1)

downstreamIdMap = pcr.downstream(lddMap, idMap)
downstreamIdArray = pcr.pcr2numpy(map = downstreamIdMap,
                                  mv = -1)
downstreamIdArray = downstreamIdArray.astype(np.int32)

ldd_df = {"downstream": downstreamIdArray[cells["y"], cells["x"]],
          "direction": lddArray[cells["y"], cells["x"]],
          "distance": distanceDownstreamArray[cells["y"], cells["x"]]}
ldd_df = pd.DataFrame(ldd_df)
ldd_df.index = cells.index
ldd_df = ldd_df.loc[ldd_df.index != ldd_df["downstream"]]

ldd_out = pl.Path("{}/ldd.csv".format(dir_out))
ldd_out.parent.mkdir(parents=True,
                     exist_ok=True)
ldd_df.to_csv(ldd_out)
