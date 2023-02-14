import pathlib as pl
import pandas as pd

from utils.load_train_test_cells import load_train_test_cells

feature_dir = pl.Path("../features/saves/global_05min")
perc_train_spat = 2 / 3
perc_train_temp = 2 / 3
dir_out = pl.Path("./saves/global_05min")
seed = 19920223

submasks = [dir.stem for dir in feature_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask), flush=True)
    
    features_submask_dir = pl.Path("{}/{}".format(feature_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))

    subsets = [dir.stem for dir in features_submask_dir.iterdir() if dir.is_dir()]

    subset = subsets[0]
    for subset in subsets:
        print("Processing {}".format(subset), flush=True)

        feature_subset_dir = pl.Path("{}/{}".format(features_submask_dir, subset))
        subset_out = pl.Path("{}/{}".format(submask_out, subset))
        
        cells_file = pl.Path("{}/cells.csv".format(feature_subset_dir))
        cells = pd.read_csv(cells_file, index_col=0)        

        cells_train_out = pl.Path("{}/cells_training.csv".format(subset_out))
        cells_val_spat_out = pl.Path("{}/cells_validation_spatial.csv".format(subset_out))
        cells_val_temp_out = pl.Path("{}/cells_validation_temporal.csv".format(subset_out))
        cells_val_spattemp_out = pl.Path("{}/cells_validation_spatiotemporal.csv".format(subset_out))

        cells_train, cells_val_spat, cells_val_temp, cells_val_spattemp = load_train_test_cells(cells=cells,
                                                                                                perc_train_spat=perc_train_spat,
                                                                                                perc_train_temp=perc_train_temp,
                                                                                                seed=seed)

        cells_train_out.parent.mkdir(parents=True,
                                    exist_ok=True)
        cells_train.to_csv(cells_train_out)

        cells_val_spat_out.parent.mkdir(parents=True,
                                        exist_ok=True)
        cells_val_spat.to_csv(cells_val_spat_out)

        cells_val_temp_out.parent.mkdir(parents=True,
                                        exist_ok=True)
        cells_val_temp.to_csv(cells_val_temp_out)

        cells_val_spattemp_out.parent.mkdir(parents=True,
                                            exist_ok=True)
        cells_val_spattemp.to_csv(cells_val_spattemp_out)
