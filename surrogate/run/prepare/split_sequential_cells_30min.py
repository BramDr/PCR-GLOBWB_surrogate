import pathlib as pl
import pandas as pd

from utils.store_sequential_cells import store_sequential_cells

feature_dir = pl.Path("../features/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/sequential")
subset_size = 2000
seed = 19920223


cells_file = pl.Path("{}/cells.csv".format(feature_dir))
cells = pd.read_csv(cells_file, index_col=0)

upstream_file = pl.Path("{}/upstream.csv".format(feature_dir))
upstream = pd.read_csv(upstream_file, index_col=0)

store_sequential_cells(cells=cells,
                       ldd=upstream,
                       subset_size=subset_size,
                       dir_out=dir_out,
                       verbose=2)
