import pathlib as pl
import re 

import numpy as np

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_transformer
from surrogate.utils.data import load_array
from surrogate.utils.data import transform_upstream_array_meta
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min/sequential")
prepare_dir = pl.Path("../prepare/saves/global_30min/sequential")
cells_file = pl.Path("../setup/saves/global_30min/cells.csv")
dir_out = pl.Path("./saves/global_30min/sequential")
datasets = ["input"]
train_subset = "train_8"

def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

# load coordinates and dates
lats = []
lons = []
dates = None

sequences = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
sequences.sort(key=sort_fn)
for sequence in sequences:
    
    prepare_sequence_dir = pl.Path("{}/{}".format(prepare_dir, sequence))
    subsets = [dir.stem for dir in prepare_sequence_dir.iterdir() if dir.is_dir()]
    subsets.sort(key=sort_fn)
    for subset in subsets:
             
        prepare_subset_dir = pl.Path("{}/{}".format(prepare_sequence_dir, subset))
        meta_file = pl.Path("{}/input/array_30-arcminute_single-year_yearly_meta.pkl".format(prepare_subset_dir))
        meta = load_meta(meta_file)
        
        lats += meta["lats"].tolist()
        lons += meta["lons"].tolist()
        dates = meta["dates"]

dataset = datasets[0]
for dataset in datasets:
    print("Processing {}".format(dataset), flush = True)
    
    dataset_out = pl.Path("{}/{}".format(dir_out, dataset))
        
    # load features
    features = []
    
    prepare_subset_dir = pl.Path("{}/../{}/cells_training/{}".format(prepare_dir, train_subset, dataset))
    meta_files = [file for file in prepare_subset_dir.rglob("*_meta.pkl")]
    for meta_file in meta_files:
        meta = load_meta(file=meta_file)
        features += np.array(meta["features"]).tolist()

    # load plots
    feature = features[0]
    for feature in features:
        print("Working on {}".format(feature), flush=True)
        
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
            
        arrays_feature=[]
        
        sequences = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
        sequences.sort(key=sort_fn)
        sequence = sequences[0]
        for sequence in sequences:

            prepare_sequence_dir = pl.Path("{}/{}".format(prepare_dir, sequence))
                        
            subsets = [dir.stem for dir in prepare_sequence_dir.iterdir() if dir.is_dir()]
            subsets.sort(key=sort_fn)
            subset = subsets[0]
            for subset in subsets:
                
                prepare_subset_dir = pl.Path("{}/{}/{}".format(prepare_sequence_dir, subset, dataset))
                
                array_files = [file for file in prepare_subset_dir.rglob("*.npy")]

                for array_file in array_files:
                    meta_file = pl.Path("{}/{}_meta.pkl".format(array_file.parent, array_file.stem))
                    meta = load_meta(file=meta_file,
                                     verbose=0)
                    
                    transformer_name = array_file.stem
                    transformer_name = re.sub(pattern="array_", repl = "transformer_", string = transformer_name)
                    transformer_file = pl.Path("{}/../{}/{}/{}.pkl".format(save_dir, train_subset, dataset, transformer_name))
                    transformer = load_transformer(file=transformer_file,
                                                   verbose=0)
                    
                    if feature not in meta["features"]:
                        continue
                    
                    array = load_array(file=array_file,
                                       meta=meta,
                                       transformer=transformer,
                                       verbose=0)
        
                    if len(array.shape) > 3:
                        array, meta = transform_upstream_array_meta(array=array,
                                                                    meta=meta,
                                                                    verbose=0)

                    feature_index = np.where(np.array(meta["features"]) == feature)[0][0]
                    array = array[:, :, [feature_index]]
                    
                    arrays_feature.append(array)
        
        array_feature = np.concatenate(arrays_feature, axis = 0)
        
        plot_df = array_to_ts_dataframe(input=array_feature,
                                        features=[feature],
                                        sequences=dates,
                                        save_sensitivity=True,
                                        save_range=True)
        plot_df.to_csv(ts_file)
        
        plot_df = array_to_cdf_dataframe(input=array_feature,
                                        features=[feature],
                                        max_values=1000)
        plot_df.to_csv(cdf_file)
        
        plot_df = array_to_sp_dataframe(input=array_feature,
                                        features=[feature],
                                        lats=lats,
                                        lons=lons,
                                        save_variability=True,
                                        save_range=True)
        plot_df.to_csv(sp_file)
