import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import pcraster as pcr

landmask_file = pl.Path(
    "../../../PCR-GLOBWB/cloneMaps/landmask_global_30min.map")
sim_time_start = dt.date(2000, 1, 1)
sim_time_end = dt.date(2010, 12, 31)
cells_out = pl.Path("./saves/global_30min/cells.csv")
seed = 19920223

pcr.setclone(str(landmask_file))
land_map = pcr.readmap(str(landmask_file))

land_map_x = pcr.ifthen(land_map, pcr.xcoordinate(land_map))
land_map_y = pcr.ifthen(land_map, pcr.ycoordinate(land_map))
land_x = pcr.pcr2numpy(land_map_x, np.nan)
land_y = pcr.pcr2numpy(land_map_y, np.nan)

lons = land_x.flatten()
lats = land_y.flatten()
xs = np.array([x for _ in range(land_y.shape[0])
              for x in range(land_x.shape[1])])
ys = np.array([y for y in range(land_y.shape[0])
              for _ in range(land_x.shape[1])])

sels = np.where(~np.isnan(lons))[0]
lons_sel = lons[sels]
lats_sel = lats[sels]
xs_sel = xs[sels]
ys_sel = ys[sels]

cells_dict = {"x": xs_sel, "y": ys_sel, "lon": lons_sel, "lat": lats_sel}
cells = pd.DataFrame(cells_dict)

cells["start"] = sim_time_start
cells["end"] = sim_time_end

cells_out.parent.mkdir(parents=True,
                       exist_ok=True)
cells.to_csv(cells_out)
