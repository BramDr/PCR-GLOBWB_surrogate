import pathlib as pl

import pandas as pd
import numpy as np
import pcraster as pcr

save_dir = pl.Path("./saves/global_05min")
ldd_file = pl.Path("../../../PCR-GLOBWB/routing/lddsound_05min.map")
dir_out = pl.Path("./saves/global_05min")

ldd = pcr.readmap(str(ldd_file))
ldd_array = pcr.pcr2numpy(map = ldd,
                         mv = 255)

submasks = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))

    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))

    cells_file = pl.Path("{}/cells.csv".format(submask_dir))
    cells = pd.read_csv(cells_file, index_col=0)
    
    id_array = np.full(shape = ldd_array.shape, fill_value=-1, dtype=np.int64)
    id_array[cells["global_y"], cells["global_x"]] = cells.index
    id = pcr.numpy2pcr(dataType = pcr.Scalar, array = id_array, mv = -1)

    downstream_id = pcr.downstream(ldd, id)
    downstream_id_array = pcr.pcr2numpy(map = downstream_id,
                                        mv = -1)
    downstream_id_array = downstream_id_array.astype(np.int32)

    ldd_df = {"downstream": downstream_id_array[cells["global_y"], cells["global_x"]]}
    ldd_df = pd.DataFrame(ldd_df)
    ldd_df.index = cells.index
    ldd_df = ldd_df.loc[ldd_df.index != ldd_df["downstream"]]

    ldd_out = pl.Path("{}/ldd.csv".format(submask_out))
    ldd_out.parent.mkdir(parents=True,
                        exist_ok=True)
    ldd_df.to_csv(ldd_out)
