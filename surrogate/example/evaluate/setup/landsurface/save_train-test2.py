import pathlib as pl

import numpy as np

from surrogate.utils.data import load_amt_dataset
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

setup_dir = pl.Path("../../../setup/landsurface/saves/train-test2")
out_dir = pl.Path("./saves/train-test2")

resolution = "30min"

setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

trainsets = [dir.stem for dir in setup_resolution_dir.iterdir() if dir.is_dir()]

trainset = trainsets[0]
for trainset in trainsets:
    print("Trainset: {}".format(trainset))

    setup_trainset_dir = pl.Path("{}/{}".format(setup_resolution_dir, trainset))
    out_trainset_dir = pl.Path("{}/{}".format(out_resolution_dir, trainset))

    datasets = [dir.stem for dir in setup_trainset_dir.iterdir() if dir.is_dir()]
    # datasets = ["input"]

    dataset = datasets[0]
    for dataset in datasets:
        print("\tDataset: {}".format(dataset))

        setup_dataset_dir = pl.Path("{}/{}".format(setup_trainset_dir, dataset))
        out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))

        features = np.unique([file.stem for file in setup_dataset_dir.rglob("*.npy") if file.is_file()])
        # features = ["routing_cropCoefficient"]

        feature = features[0]
        for feature in features:
            print("\t\tFeature: {}".format(feature))

            ts_file = pl.Path("{}/{}_ts.csv".format(out_dataset_dir, feature))
            sp_file = pl.Path("{}/{}_sp.csv".format(out_dataset_dir, feature))
            if ts_file.exists() and sp_file.exists():
                print("Already done")
                continue

            array, meta, _ = load_amt_dataset(feature=feature,
                                                array_dir=setup_dataset_dir,
                                                meta_dir=setup_dataset_dir,
                                                verbose=0)

            if meta is None:
                raise ValueError("Metas are None")

            plot_df = array_to_ts_dataframe(input=array,
                                            features=meta["features"],
                                            sequences=meta["dates"],
                                            save_sensitivity=True,
                                            save_range=False)
            ts_file.parent.mkdir(parents=True, exist_ok=True)
            plot_df.to_csv(ts_file)

            plot_df = array_to_sp_dataframe(input=array,
                                            features=meta["features"],
                                            lats=meta["lats"],
                                            lons=meta["lons"],
                                            save_variability=True,
                                            save_range=False)
            sp_file.parent.mkdir(parents=True, exist_ok=True)
            plot_df.to_csv(sp_file)
