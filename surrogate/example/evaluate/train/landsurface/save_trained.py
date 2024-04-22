import pathlib as pl

import numpy as np

from surrogate.utils.data import load_amt_dataset
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

train_dir = pl.Path("../../../train/landsurface/saves/train-test2")
out_dir = pl.Path("./saves")
dataset = "output"

resolution = "30min"

train_resolution_dir = pl.Path("{}/{}".format(train_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

trainsets = [dir.stem for dir in train_resolution_dir.iterdir() if dir.is_dir()]

trainset = trainsets[0]
for trainset in trainsets:
    print("Trainset: {}".format(trainset))

    train_trainset_dir = pl.Path("{}/{}".format(train_resolution_dir, trainset))
    out_trainset_dir = pl.Path("{}/{}".format(out_resolution_dir, trainset))
    
    train_dataset_dir = pl.Path("{}/{}".format(train_trainset_dir, dataset))
    out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))
    
    features = [file.stem for file in train_dataset_dir.glob("*.npy") if file.is_file()]
    
    feature_index = 0
    feature = features[0]
    for feature_index, feature in enumerate(features):
        print("\tFeature: {}".format(feature))

        ts_file = pl.Path("{}/{}_ts.csv".format(out_dataset_dir, feature))
        sp_file = pl.Path("{}/{}_sp.csv".format(out_dataset_dir, feature))
        if ts_file.exists() and sp_file.exists():
            print("Already done")
            continue
        
        array, meta, _ = load_amt_dataset(feature=feature,
                                            array_dir=train_dataset_dir,
                                            meta_dir=train_dataset_dir,
                                            verbose=0)
        
        if meta is None:
            raise ValueError("Meta is None")
    
        plot_df = array_to_ts_dataframe(input=array,
                                        features=np.array([feature]),
                                        sequences=meta["dates"],
                                        save_sensitivity=True,
                                        save_range=False)
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(ts_file)

        plot_df = array_to_sp_dataframe(input=array,
                                        features=np.array([feature]),
                                        lats=meta["lats"],
                                        lons=meta["lons"],
                                        save_variability=True,
                                        save_range=False)
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(sp_file)