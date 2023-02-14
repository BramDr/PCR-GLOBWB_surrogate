import pathlib as pl
import pandas as pd

from utils.store_train_test_cells import store_train_test_cells

feature_dir = pl.Path("../features/saves/global_30min")
perc_train_spat = 2 / 3
perc_train_temp = 2 / 3
dir_out = pl.Path("./saves/global_30min")
seed = 19920223

subsets = [dir.stem for dir in feature_dir.iterdir() if dir.is_dir()]

subset = subsets[0]
for subset in subsets:
    print("Working on {}".format(subset), flush=True)

    feature_subset_dir = pl.Path("{}/{}".format(feature_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))
    
    cells_file = pl.Path("{}/cells.csv".format(feature_subset_dir))
    cells = pd.read_csv(cells_file, index_col=0)

    store_train_test_cells(cells=cells,
                           perc_train_spat=perc_train_spat,
                           perc_train_temp=perc_train_temp,
                           dir_out=subset_out,
                           seed=seed)
