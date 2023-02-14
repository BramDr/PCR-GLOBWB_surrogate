import pathlib as pl

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_array
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min")
dir_out = pl.Path("./saves/global_30min")
datasets = ["input", "output"]

dataset = datasets[0]
for dataset in datasets:
    print("Working on {}".format(dataset))

    dataset_dir = pl.Path("{}/{}".format(save_dir, dataset))
    dataset_out = pl.Path("{}/{}".format(dir_out, dataset))

    array_files = [file for file in dataset_dir.rglob("*.npy")]

    array_file = array_files[0]
    for array_file in array_files:
        
        feature = array_file.stem
        section = [parent.stem for parent in array_file.parents if "Options" in parent.stem]
        if len(section) > 0:
            feature = "{}_{}".format(section[0], feature)

        print("Processing {}".format(feature))

        ts_file = pl.Path("{}_{}_ts.csv".format(dataset_out, feature))
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        cdf_file = pl.Path("{}_{}_cdf.csv".format(dataset_out, feature))
        cdf_file.parent.mkdir(parents=True, exist_ok=True)
        sp_file = pl.Path("{}_{}_sp.csv".format(dataset_out, feature))
        sp_file.parent.mkdir(parents=True, exist_ok=True)
        
        if ts_file.exists() and cdf_file.exists() and sp_file.exists():
            continue
        
        meta_file = pl.Path("{}/{}_meta.pkl".format(array_file.parent, array_file.stem))
        meta = load_meta(file=meta_file)

        array = load_array(file=array_file,
                           meta=meta)

        plot_df = array_to_ts_dataframe(input=array,
                                        features=[feature],
                                        sequences=meta["dates"],
                                        save_sensitivity=True)
        plot_df.to_csv(ts_file)

        plot_df = array_to_cdf_dataframe(input=array,
                                         features=[feature],
                                         max_values=1000)
        plot_df.to_csv(cdf_file)

        plot_df = array_to_sp_dataframe(input=array,
                                        features=[feature],
                                        lats=meta["lats"],
                                        lons=meta["lons"],
                                        save_variability=True,
                                        save_range=True)
        plot_df.to_csv(sp_file)
