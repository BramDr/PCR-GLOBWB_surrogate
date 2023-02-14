import pathlib as pl
import numpy as np

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_array
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min/sequential")
prepare_dir = pl.Path("../prepare/saves/global_30min/sequential")
dir_out = pl.Path("./saves/global_30min/sequential")
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
    
    sequence_dir = pl.Path("{}/{}".format(prepare_dir, sequence))
    subsets = [dir.stem for dir in sequence_dir.iterdir() if dir.is_dir()]
    subsets.sort(key=sort_fn)
    for subset in subsets:
             
        subset_dir = pl.Path("{}/{}".format(sequence_dir, subset))
        meta_file = pl.Path("{}/input/array_30-arcminute_single-year_yearly_meta.pkl".format(subset_dir))
        meta = load_meta(meta_file)
        
        lats += meta["lats"].tolist()
        lons += meta["lons"].tolist()
        dates = meta["dates"]

# load features
features = []

prepare_subset_dir = pl.Path("{}/../{}/cells_training/output".format(prepare_dir, train_subset))

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
    
    ts_file = pl.Path("{}/predicted_{}_ts.csv".format(dir_out, feature))
    ts_file.parent.mkdir(parents=True, exist_ok=True)
    cdf_file = pl.Path("{}/predicted_{}_cdf.csv".format(dir_out, feature))
    cdf_file.parent.mkdir(parents=True, exist_ok=True)
    sp_file = pl.Path("{}/predicted_{}_sp.csv".format(dir_out, feature))
    sp_file.parent.mkdir(parents=True, exist_ok=True)
    
    arrays_feature=[]
    
    sequences = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
    sequences.sort(key=sort_fn)
    sequence = sequences[0]
    for sequence in sequences:

        sequence_dir = pl.Path("{}/{}".format(save_dir, sequence))
        subsets = [dir.stem for dir in sequence_dir.iterdir() if dir.is_dir()]
        subsets.sort(key=sort_fn)
        subset = subsets[0]
        for subset in subsets:
            
            subset_dir = pl.Path("{}/{}".format(sequence_dir, subset))
            array_file = pl.Path("{}/predicted.npy".format(subset_dir))
            
            array = load_array(file=array_file,
                               verbose = 0)
            
            feature_index = np.where(np.array(features) == feature)[0][0]
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
