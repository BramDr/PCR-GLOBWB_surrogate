import pathlib as pl

import pandas as pd

from surrogate.utils.data import load_concatenate_arrays_metas
from surrogate.utils.data import combine_spatiotemporal_arrays_metas
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    prepare_submask_dir = pl.Path("{}/{}".format(prepare_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))
    
    trainsets = [dir.stem for dir in submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

    trainset = "train_96"
    for trainset in trainsets:
        print("Working on {}".format(trainset), flush=True)

        trainset_dir = pl.Path("{}/{}".format(submask_dir, trainset))
        prepare_trainset_dir = pl.Path("{}/{}".format(prepare_submask_dir, trainset))
        trainset_out = pl.Path("{}/{}".format(submask_out, trainset))

        subset_dirs = [pl.Path("{}/cells_training".format(prepare_trainset_dir)),
                        pl.Path("{}/cells_validation_spatial".format(prepare_trainset_dir)),
                        pl.Path("{}/cells_validation_temporal".format(prepare_trainset_dir)),
                        pl.Path("{}/cells_validation_spatiotemporal".format(prepare_trainset_dir))]

        dataset = datasets[0]
        for dataset in datasets:
            print("Processing {}".format(dataset), flush = True)
            
            if dataset == "input":
                features = in_features
            elif dataset == "output":
                features = out_features
            else:
                raise ValueError("Dataset {} could not be processed".format(dataset))

            ts_file = pl.Path("{}/{}_ts.csv".format(trainset_out, dataset))
            ts_file.parent.mkdir(parents=True, exist_ok=True)
            cdf_file = pl.Path("{}/{}_cdf.csv".format(trainset_out, dataset))
            cdf_file.parent.mkdir(parents=True, exist_ok=True)
            sp_file = pl.Path("{}/{}_sp.csv".format(trainset_out, dataset))
            sp_file.parent.mkdir(parents=True, exist_ok=True)

            if ts_file.exists() and cdf_file.exists() and sp_file.exists():
                continue

            arrays, metas = load_concatenate_arrays_metas(save_dirs=subset_dirs,
                                                          dataset=dataset,
                                                          transformer_dir=trainset_dir,
                                                          features=features["feature"],
                                                          verbose=0)
                
            array, meta = combine_spatiotemporal_arrays_metas(train = arrays[0],
                                                    train_meta = metas[0],
                                                    spatial_test= arrays[1],
                                                    spatial_test_meta=metas[1],
                                                    temporal_test= arrays[2],
                                                    temporal_test_meta=metas[2],
                                                    spatiotemporal_test= arrays[3],
                                                    spatiotemporal_test_meta= metas[3],
                                                    verbose=2)

            plot_df = array_to_ts_dataframe(input=array,
                                            features=meta["features"],
                                            sequences=meta["dates"],
                                            save_sensitivity=True,
                                            save_range=True)
            plot_df.to_csv(ts_file)

            plot_df = array_to_cdf_dataframe(input=array,
                                            features=meta["features"],
                                            max_values=1000)
            plot_df.to_csv(cdf_file)
                    
            plot_df = array_to_sp_dataframe(input=array,
                                            features=meta["features"],
                                            lats=meta["lats"],
                                            lons=meta["lons"]
    ,                                       save_variability=True,
                                            save_range=True)
            plot_df.to_csv(sp_file)
