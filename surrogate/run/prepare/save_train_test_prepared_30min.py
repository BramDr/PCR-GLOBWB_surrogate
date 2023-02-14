import pathlib as pl

from surrogate.utils.data import load_concatenate_arrays_metas
from surrogate.utils.data import combine_spatiotemporal_arrays_metas
from surrogate.utils.data import transform_upstream_arrays_metas
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min")
dir_out = pl.Path("./saves/global_30min")
datasets = ["input", "output", "upstream_input", "upstream_output", "upstream_details"]
datasets = ["upstream_input", "upstream_details"]

subsets = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

subset = "train_96"
for subset in subsets:
    print("Working on {}".format(subset), flush=True)

    subset_dir = pl.Path("{}/{}".format(save_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))

    train_test_dirs = [pl.Path("{}/cells_training".format(subset_dir)),
                       pl.Path("{}/cells_validation_spatial".format(subset_dir)),
                       pl.Path("{}/cells_validation_temporal".format(subset_dir)),
                       pl.Path("{}/cells_validation_spatiotemporal".format(subset_dir))]

    dataset = datasets[0]
    for dataset in datasets:
        print("Processing {}".format(dataset), flush = True)

        ts_file = pl.Path("{}/{}_ts.csv".format(subset_out, dataset))
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        cdf_file = pl.Path("{}/{}_cdf.csv".format(subset_out, dataset))
        cdf_file.parent.mkdir(parents=True, exist_ok=True)
        sp_file = pl.Path("{}/{}_sp.csv".format(subset_out, dataset))
        sp_file.parent.mkdir(parents=True, exist_ok=True)

        #if ts_file.exists() and cdf_file.exists():
        #    continue

        arrays, metas = load_concatenate_arrays_metas(save_dirs=train_test_dirs,
                                                      dataset=dataset,
                                                      verbose=1)
        
        if len(arrays[0].shape) > 3:
            arrays, metas = transform_upstream_arrays_metas(arrays=arrays,
                                                            metas=metas,
                                                            verbose=1)
        
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
