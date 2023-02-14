import pandas as pd
import pathlib as pl

from utils.store_self import store_self

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
setup_dir = pl.Path("../setup/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]
datasets = ["output"]

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")
        
def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

submasks = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}/sequential".format(save_dir, submask))
    setup_submask_dir = pl.Path("{}/{}".format(setup_dir, submask))
    submask_out = pl.Path("{}/{}/sequential".format(dir_out, submask))

    dataset = datasets[0]
    for dataset in datasets:
        print("Processing {}".format(dataset), flush = True)

        if dataset == "input":
            features = in_features
        elif dataset == "output":
            features = out_features
            features = features.loc[features["feature"] == "discharge"]
        else:
            raise ValueError("dataset {} cannot be processed".format(dataset))

        name_list = []
        samples_list = []
        dates_list = []
        lats_list = []
        lons_list = []
        dir_out_list = []
        
        sequences = [dir.stem for dir in submask_dir.iterdir() if dir.is_dir()]
        sequences.sort(key=sort_fn)

        sequence = sequences[0]
        for sequence in sequences:
            sequence_dir = pl.Path("{}/{}".format(submask_dir, sequence))
            sequence_out = pl.Path("{}/{}".format(submask_out, sequence))

            subset_files = [file for file in sequence_dir.iterdir() if not file.is_dir() and file.suffix == ".csv"]
            
            subset_file = subset_files[0]
            for subset_file in subset_files:
                subset_out = pl.Path("{}/{}/{}".format(sequence_out, subset_file.stem, dataset))

                cells = pd.read_csv(subset_file, index_col=0)
            
                lats = cells["lat"].to_numpy()
                lons = cells["lon"].to_numpy()
                samples = cells.index.to_numpy()
                dates = pd.date_range(start = cells["start"].iloc[0],
                                    end = cells["end"].iloc[0],
                                    freq="D").to_pydatetime()
                
                name_list.append("{} {} {}".format(submask, sequence, subset_file.stem))
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