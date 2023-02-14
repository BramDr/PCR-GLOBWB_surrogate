import pathlib as pl

import pandas as pd

from surrogate.utils.data import load_concatenate_arrays_metas
from surrogate.utils.data import concatenate_arrays_metas
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
train_subset = "train_32"
datasets = ["input"]

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

submasks = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    prepare_submask_dir = pl.Path("{}/{}/sequential".format(prepare_dir, submask))
    submask_out = pl.Path("{}/{}/sequential".format(dir_out, submask))

    trainset_dir = pl.Path("{}/{}".format(submask_dir, train_subset))
    
    subset_dirs = []
    
    sequences = [dir.stem for dir in prepare_submask_dir.iterdir() if dir.is_dir()]
    sequences.sort(key=sort_fn)
    sequence = sequences[0]
    for sequence in sequences:
        prepare_sequence_dir = pl.Path("{}/{}".format(prepare_submask_dir, sequence))
        subsets = [dir.stem for dir in prepare_sequence_dir.iterdir() if dir.is_dir()]
        subsets.sort(key=sort_fn)
        subset = subsets[0]
        for subset in subsets:
            prepare_subset_dir = pl.Path("{}/{}".format(prepare_sequence_dir, subset))
            subset_dirs.append(prepare_subset_dir)

    dataset = datasets[0]
    for dataset in datasets:
        print("Working on {}".format(dataset), flush = True)
            
        if dataset == "input":
            features = in_features
        elif dataset == "output":
            features = out_features
        else:
            raise ValueError("Dataset {} could not be processed".format(dataset))

        dataset_out = pl.Path("{}/{}".format(submask_out, dataset))
        
        feature = features["feature"].iloc[0]
        for feature in features["feature"]:

            print("Processing {}".format(feature), flush = True)
            
            ts_file = pl.Path("{}_{}_ts.csv".format(dataset_out, feature))
            ts_file.parent.mkdir(parents=True, exist_ok=True)
            cdf_file = pl.Path("{}_{}_cdf.csv".format(dataset_out, feature))
            cdf_file.parent.mkdir(parents=True, exist_ok=True)
            sp_file = pl.Path("{}_{}_sp.csv".format(dataset_out, feature))
            sp_file.parent.mkdir(parents=True, exist_ok=True)
            
            if ts_file.exists() and cdf_file.exists() and sp_file.exists():
                continue
            
            arrays, metas = load_concatenate_arrays_metas(save_dirs=subset_dirs,
                                                          dataset=dataset,
                                                          transformer_dir = trainset_dir,
                                                          features=[feature],
                                                          verbose=0)
            
            array, meta = concatenate_arrays_metas(arrays=arrays,
                                                   metas=metas,
                                                   direction="sample",
                                                   verbose=1)
            
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
                                            lons=meta["lons"],
                                            save_variability=True,
                                            save_range=True)
            plot_df.to_csv(sp_file)
