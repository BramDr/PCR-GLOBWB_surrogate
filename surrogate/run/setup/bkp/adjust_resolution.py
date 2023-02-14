import pathlib as pl
import pickle

from surrogate.utils.data import load_meta

save_dir = pl.Path("./saves")

meta_files = [file for file in save_dir.rglob("*_meta.pkl")]

meta_file = meta_files[0]
for meta_file in meta_files:
    meta = load_meta(meta_file)
    
    resolution = meta["x_resolution"]    
    number = int(resolution.split("-")[-2])
    name = resolution.split("-")[1]
    resolution = "{:02d}-{}".format(number, name)
    meta["x_resolution"] = resolution
    
    resolution = meta["y_resolution"]
    number = int(resolution.split("-")[-2])
    name = resolution.split("-")[1]
    resolution = "{:02d}-{}".format(number, name)
    meta["y_resolution"] = resolution
    
    with open(meta_file, "wb") as handle:
        pickle.dump(meta, handle)
    