import pathlib as pl

import pandas as pd

from utils.store_self import store_self

save_dir = pl.Path("./saves/global_05min")
setup_dir = pl.Path("../setup/saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    setup_submask_dir = pl.Path("{}/{}".format(setup_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))

    dataset = datasets[0]
    for dataset in datasets:
        print("Processing {}".format(dataset), flush = True)

        if dataset == "input":
            features = in_features
        elif dataset == "output":
            features = out_features
        else:
            raise ValueError("dataset {} cannot be processed".format(dataset))
        
        name_list = []
        samples_list = []
        dates_list = []
        lats_list = []
        lons_list = []
        dir_out_list = []

        trainsets = [dir.stem for dir in submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

        trainset = trainsets[0]
        for trainset in trainsets:

            trainset_dir = pl.Path("{}/{}".format(submask_dir, trainset))
            trainset_out = pl.Path("{}/{}".format(submask_out, trainset))

            subset_files = [pl.Path("{}/cells_training.csv".format(trainset_dir)),
                            pl.Path("{}/cells_validation_spatial.csv".format(trainset_dir)),
                            pl.Path("{}/cells_validation_temporal.csv".format(trainset_dir)),
                            pl.Path("{}/cells_validation_spatiotemporal.csv".format(trainset_dir))]

            subset_file = subset_files[0]
            for subset_file in subset_files:

                subset_out = pl.Path("{}/{}/{}".format(trainset_out, subset_file.stem, dataset))

                cells = pd.read_csv(subset_file, index_col=0)
            
                lats = cells["lat"].to_numpy()
                lons = cells["lon"].to_numpy()
                samples = cells.index.to_numpy()
                dates = pd.date_range(start = cells["start"].iloc[0],
                                    end = cells["end"].iloc[0],
                                    freq="D").to_pydatetime()
                
                name_list.append("{} {}".format(submask, subset_file.stem))
                samples_list.append(samples)
                dates_list.append(dates)
                lats_list.append(lats)
                lons_list.append(lons)
                dir_out_list.append(subset_out)

        store_self(save_dir=setup_submask_dir,
                   features=features,
                   resolution="05-arcminute",
                   name_list=name_list,
                   samples_list=samples_list,
                   dates_list=dates_list,
                   lats_list=lats_list,
                   lons_list=lons_list,
                   dir_out_list=dir_out_list,
                   verbose=1)