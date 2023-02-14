import pathlib as pl

import pandas as pd
import numpy as np

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_array
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))
    
    dataset = datasets[0]
    for dataset in datasets:
        print("Working on {}".format(dataset))
        
        if dataset == "input":
            features = in_features
        elif dataset == "output":
            features = out_features
        else:
            raise ValueError("Dataset {} could not be processed".format(dataset))

        dataset_out = pl.Path("{}/{}".format(submask_out, dataset))

        index = features.index[0]
        for index in features.index:
            
            source = features["source"].loc[index]
            feature = features["feature"].loc[index]
            section = features["section"].loc[index]
            option = features["option"].loc[index]
            variable = features["variable"].loc[index]
            constant = features["constant"].loc[index]

            print("Processing {}".format(feature))

            ts_file = pl.Path("{}_{}_ts.csv".format(dataset_out, feature))
            ts_file.parent.mkdir(parents=True, exist_ok=True)
            cdf_file = pl.Path("{}_{}_cdf.csv".format(dataset_out, feature))
            cdf_file.parent.mkdir(parents=True, exist_ok=True)
            sp_file = pl.Path("{}_{}_sp.csv".format(dataset_out, feature))
            sp_file.parent.mkdir(parents=True, exist_ok=True)
            
            if ts_file.exists() and cdf_file.exists() and sp_file.exists():
                continue
            
            if len(variable) == 0:
                continue
            
            setup_subset_dir = pl.Path("{}/{}/{}/{}".format(submask_dir, source, section, option))
            
            meta_file = pl.Path("{}/{}_meta.pkl".format(setup_subset_dir, variable))
            meta = load_meta(file=meta_file)
            
            array_file = pl.Path("{}/{}.npy".format(setup_subset_dir, variable))
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
