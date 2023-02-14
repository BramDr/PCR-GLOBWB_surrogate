import pandas as pd
import pathlib as pl

from utils.data.get_upstream_samples import get_upstream_samples
from utils.data.get_downstream_samples import get_downstream_samples
from utils.store_self import store_self
from utils.store_upstream import store_upstream
from utils.store_downstream import store_downstream

save_dir = pl.Path("./saves/global_30min")
setup_dir = pl.Path("../setup/saves/global_30min")
feature_dir = pl.Path("../features/saves/global_30min")
dir_out = pl.Path("./saves/global_30min")

self_input_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
self_output_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
upstream_input_features_file = pl.Path("{}/features_upstream_input.csv".format(feature_dir))
upstream_output_features_file = pl.Path("{}/features_upstream_output.csv".format(feature_dir))
downstream_input_features_file = pl.Path("{}/features_downstream_input.csv".format(feature_dir))
downstream_output_features_file = pl.Path("{}/features_downstream_output.csv".format(feature_dir))
self_input_features = pd.read_csv(self_input_features_file, keep_default_na=False).fillna("")
self_output_features = pd.read_csv(self_output_features_file, keep_default_na=False).fillna("")
upstream_input_features = pd.read_csv(upstream_input_features_file, keep_default_na=False).fillna("")
upstream_output_features = pd.read_csv(upstream_output_features_file, keep_default_na=False).fillna("")
downstream_input_features = pd.read_csv(downstream_input_features_file, keep_default_na=False).fillna("")
downstream_output_features = pd.read_csv(downstream_output_features_file, keep_default_na=False).fillna("")

upstream_file = pl.Path("{}/upstream.csv".format(feature_dir))
upstream = pd.read_csv(upstream_file, index_col=0)
upstream = upstream[upstream["from"] != upstream["to"]]

input_dir = pl.Path("{}/input".format(setup_dir))
output_dir = pl.Path("{}/output".format(setup_dir))

trainsets = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

trainset = "hyper"
for trainset in trainsets:
    print("Working on {}".format(trainset), flush=True)

    trainset_dir = pl.Path("{}/{}".format(save_dir, trainset))
    trainset_out = pl.Path("{}/{}".format(dir_out, trainset))

    subset_files = [pl.Path("{}/cells_training.csv".format(trainset_dir)),
                    pl.Path("{}/cells_validation_spatial.csv".format(trainset_dir)),
                    pl.Path("{}/cells_validation_temporal.csv".format(trainset_dir)),
                    pl.Path("{}/cells_validation_spatiotemporal.csv".format(trainset_dir))]

    subset_file = subset_files[0]
    for subset_file in subset_files:
        print("Working on {}".format(subset_file.stem), flush=True)

        subset_out = pl.Path("{}/{}".format(trainset_out, subset_file.stem))

        cells = pd.read_csv(subset_file, index_col=0)
    
        lats = cells["lat"].to_numpy()
        lons = cells["lon"].to_numpy()
        samples = cells.index.to_numpy()
        dates = pd.date_range(start = cells["start"].iloc[0],
                            end = cells["end"].iloc[0],
                            freq="D").to_pydatetime()
        
        upstream_samples = get_upstream_samples(upstream=upstream,
                                                samples=samples)
        
        downstream_samples = get_downstream_samples(upstream=upstream,
                                                    samples=upstream_samples)

        #store_self(input_dir=input_dir,
        #           output_dir=output_dir,
        #           input_features=self_input_features,
        #           output_features=self_output_features,
        #           samples=samples,
        #           dates=dates,
        #           lats=lats,
        #           lons=lons,
        #           resolution="30-arcminute",
        #           dir_out=subset_out,
        #           verbose = 2)

        store_upstream(input_dir=input_dir,
                        output_dir=output_dir,
                        upstream=upstream,
                        input_features=upstream_input_features,
                        output_features=upstream_output_features,
                        samples=samples,
                        upstream_samples=upstream_samples,
                        dates=dates,
                        lats=lats,
                        lons=lons,
                        resolution="30-arcminute",
                        dir_out=subset_out,
                        verbose = 2)

        #store_downstream(input_dir=input_dir,
        #                output_dir=output_dir,
        #                downstream=upstream,
        #                input_features=downstream_input_features,
        #                output_features=downstream_output_features,
        #                samples=samples,
        #                upstream_samples=upstream_samples,
        #                downstream_samples=downstream_samples,
        #                dates=dates,
        #                lats=lats,
        #                lons=lons,
        #                resolution="30-arcminute",
        #                dir_out=subset_out,
        #                verbose = 2)