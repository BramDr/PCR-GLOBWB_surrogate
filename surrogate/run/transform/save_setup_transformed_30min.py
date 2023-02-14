import pathlib as pl
import numpy as np

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_metas
from surrogate.utils.data import load_transformer
from surrogate.utils.data import load_array
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min")
setup_dir = pl.Path("../setup/saves/global_30min")
prepare_dir = pl.Path("../prepare/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/setup")
subset = "train_8"
datasets = ["input", "output"]

subset_dir = pl.Path("{}/{}".format(save_dir, subset))
prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))

transformer_meta_files = [file for file in prepare_subset_dir.rglob("*_meta.pkl")]
transformer_metas = load_metas(files=transformer_meta_files)

dataset = datasets[0]
for dataset in datasets:
    print("Working on {}".format(dataset))

    setup_dataset_dir = pl.Path("{}/{}".format(setup_dir, dataset))
    dataset_out = pl.Path("{}/{}".format(dir_out, dataset))

    array_files = [file for file in setup_dataset_dir.rglob("*.npy")]
    
    array_file = array_files[0]
    for array_file in array_files:
        
        feature = array_file.stem
        section = [parent.stem for parent in array_file.parents if "Options" in parent.stem]
        if len(section) > 0:
            feature = "{}:{}".format(section[0], feature)
        
        print("Processing {}".format(feature))
            
        feature_sections = feature.split(":")
        feature_name = "-".join(feature_sections)
        
        ts_file = pl.Path("{}_{}_ts.csv".format(dataset_out, feature_name))
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        cdf_file = pl.Path("{}_{}_cdf.csv".format(dataset_out, feature_name))
        cdf_file.parent.mkdir(parents=True, exist_ok=True)
        sp_file = pl.Path("{}_{}_sp.csv".format(dataset_out, feature_name))
        sp_file.parent.mkdir(parents=True, exist_ok=True)

        if ts_file.exists() and cdf_file.exists() and sp_file.exists():
            continue
        
        meta_file = pl.Path("{}/{}_meta.pkl".format(array_file.parent, array_file.stem))
        meta = load_meta(file=meta_file)
        
        frequency = meta["date_frequency"]
        resolution = meta["x_resolution"]
            
        transformer_file = pl.Path("{}/{}/array_{}_{}_transformer.pkl".format(subset_dir, dataset, resolution, frequency))
        transformer = load_transformer(transformer_file)
        
        transformer_meta_file = pl.Path("{}/cells_training/{}/array_{}_{}_meta.pkl".format(prepare_subset_dir, dataset, resolution, frequency))
        transformer_meta = load_meta(transformer_meta_file)
        index = np.where(transformer_meta["features"] == feature)[0]
        transformer = transformer.subset(index)

        array = load_array(file=array_file,
                           meta=meta,
                           transformer=transformer)

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
