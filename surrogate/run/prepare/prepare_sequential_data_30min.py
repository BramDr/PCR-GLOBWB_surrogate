import pandas as pd
import pathlib as pl

from utils.store_upstream import store_upstream
from utils.store_self import store_self

save_dir = pl.Path("./saves/global_30min/sequential")
feature_dir = pl.Path("../features/saves/global_30min")
setup_dir = pl.Path("../setup/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/sequential")

self_input_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
self_output_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
upstream_input_features_file = pl.Path("{}/features_upstream_input.csv".format(feature_dir))
upstream_output_features_file = pl.Path("{}/features_upstream_output.csv".format(feature_dir))
self_input_features = pd.read_csv(self_input_features_file, keep_default_na=False).fillna("")
self_output_features = pd.read_csv(self_output_features_file, keep_default_na=False).fillna("")
upstream_input_features = pd.read_csv(upstream_input_features_file, keep_default_na=False).fillna("")
upstream_output_features = pd.read_csv(upstream_output_features_file, keep_default_na=False).fillna("")

def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

upstream_file = pl.Path("{}/upstream_cd.csv".format(feature_dir))
upstream = pd.read_csv(upstream_file, index_col=0)

input_dir = pl.Path("{}/input".format(setup_dir))
output_dir = pl.Path("{}/output".format(setup_dir))

sequences = [dir.stem for dir in dir_out.iterdir() if dir.is_dir()]
sequences.sort(key=sort_fn)

sequence = sequences[0]
for sequence in sequences:
    print("Working on {}".format(sequence), flush=True)

    sequence_dir = pl.Path("{}/{}".format(save_dir, sequence))
    sequence_out = pl.Path("{}/{}".format(dir_out, sequence))

    subset_files = [file for file in sequence_dir.iterdir() if not file.is_dir() and file.suffix == ".csv"]

    for subset_file in subset_files:
        print("Working on {}".format(subset_file.stem), flush=True)

        subset_out = pl.Path("{}/{}".format(dir_out, subset_file.stem))

        cells = pd.read_csv(subset_file, index_col=0)
    
        lats = cells["lat"].to_numpy()
        lons = cells["lon"].to_numpy()
        samples = cells.index.to_numpy()
        dates = pd.date_range(start = cells["start"].iloc[0],
                            end = cells["end"].iloc[0],
                            freq="D").to_pydatetime()

        store_self(dataset_dirs=input_dir,
                output_dir=output_dir,
                store_output=False,
                dataset_features=self_input_features,
                output_features=self_output_features,
                samples=samples,
                dates=dates,
                lats=lats,
                lons=lons,
                resolution="30-arcminute",
                dir_out=subset_out)

        store_upstream(input_dir=input_dir,
                        output_dir=output_dir,
                        upstream=upstream,
                        store_output=False,
                        input_features=upstream_input_features,
                        output_features=upstream_output_features,
                        samples=samples,
                        dates=dates,
                        lats=lats,
                        lons=lons,
                        resolution="30-arcminute",
                        dir_out=subset_out)
