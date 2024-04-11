import pathlib as pl
import datetime as dt

import pandas as pd

base_dir = pl.Path("../saves")
out_dir = pl.Path("./saves/train-test2")
seed = 19920223
time_start = dt.date(2000, 1, 1)
time_end = dt.date(2010, 12, 31)

base_fraction = 1 / 8
resolution_fractions = {"30min": 1 / 1,
                        "05min": 1 / 16,
                        "30sec": 1 / 64}
test_fraction = 1 / 3
validate_fraction = 1 / 4
temporal_fraction = 2 / 3

time_split = time_start + (time_end - time_start) * temporal_fraction

resolutions = ["30min", "05min", "30sec"]

resolution = resolutions[-1]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))

    base_resolution_dir = pl.Path("{}/{}".format(base_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    cells_file = pl.Path("{}/cells.parquet".format(base_resolution_dir))
    cells = pd.read_parquet(cells_file)
    
    resolution_fraction = resolution_fractions[resolution]
    
    subset_nsamples = int(cells.index.size * base_fraction * resolution_fraction)
    test_nsamples = int(subset_nsamples * test_fraction)
    train_nsamples = subset_nsamples - test_nsamples
    train_validate_nsamples = int(train_nsamples * validate_fraction)
    train_train_nsamples = train_nsamples - train_validate_nsamples
    
    subset_cells = cells.sample(n = subset_nsamples,
                                random_state=seed,
                                axis=0)
    
    subset_cells_out = pl.Path("{}/cells.parquet".format(out_resolution_dir))
    subset_cells_out.parent.mkdir(parents=True, exist_ok=True)
    subset_cells.to_parquet(subset_cells_out)
    
    train_cells = subset_cells.sample(n = train_train_nsamples,
                                      random_state=seed,
                                      axis=0)
    subset_cells = subset_cells.drop(train_cells.index,
                                        axis=0)
    train_cells = train_cells.sort_index()
    train_cells["start"] = time_start
    train_cells["end"] = time_split
    
    validate_cells = subset_cells.sample(n = train_validate_nsamples,
                                         random_state=seed,
                                         axis=0)
    subset_cells = subset_cells.drop(validate_cells.index,
                                        axis=0)
    validate_cells = validate_cells.sort_index()
    validate_cells["start"] = time_start
    validate_cells["end"] = time_split
    
    test_cells = subset_cells.sample(n = test_nsamples,
                                     random_state=seed,
                                     axis=0)
    subset_cells = subset_cells.drop(test_cells.index,
                                        axis=0)
    test_cells = test_cells.sort_index()
    test_cells["start"] = time_start
    test_cells["end"] = time_end
    
    print("samples {}: train {}, validate {} and test {}".format(subset_nsamples,
                                                                 train_cells.index.size,
                                                                 validate_cells.index.size,
                                                                 test_cells.index.size))
    
    train_cells_out = pl.Path("{}/train/cells.parquet".format(out_resolution_dir))
    train_cells_out.parent.mkdir(parents=True, exist_ok=True)
    train_cells.to_parquet(train_cells_out)
    
    validate_cells_out = pl.Path("{}/validate/cells.parquet".format(out_resolution_dir))
    validate_cells_out.parent.mkdir(parents=True, exist_ok=True)
    validate_cells.to_parquet(validate_cells_out)
    
    test_cells_out = pl.Path("{}/test/cells.parquet".format(out_resolution_dir))
    test_cells_out.parent.mkdir(parents=True, exist_ok=True)
    test_cells.to_parquet(test_cells_out)
