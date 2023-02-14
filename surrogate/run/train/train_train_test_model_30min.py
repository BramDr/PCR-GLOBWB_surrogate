import pathlib as pl
import pandas as pd

from surrogate.utils.data import load_batchsets
from surrogate.utils.data import load_batchsets
from utils.build_train_model import build_loader_model_trainer

prepare_dir = pl.Path("../prepare/saves/global_30min")
transformer_dir = pl.Path("../transform/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min")
seed = 19920232
samples_size = 32
dates_size = 365
n_backwards = 2e4

upstream = pd.read_csv(upstream_file, index_col=0)

subsets = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

subset = "train_96"
for subset in subsets:
    print("Working on {}".format(subset), flush = True)

    prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
    transformer_subset_dir = pl.Path("{}/{}".format(transformer_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))

    train_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
    test_dirs = [pl.Path("{}/cells_validation_spatial".format(prepare_subset_dir)),
                    pl.Path("{}/cells_validation_temporal".format(prepare_subset_dir)),
                    pl.Path("{}/cells_validation_spatiotemporal".format(prepare_subset_dir))]

    train_dataset = load_batchsets(save_dir=train_dir,
                                transformer_dir=transformer_subset_dir,
                                upstream=upstream,
                                permute=True,
                                seed=seed,
                                sample_size=samples_size,
                                dates_size=dates_size,
                                verbose=1)
    
    test_datasets = load_batchsets(save_dirs=test_dirs,
                                transformer_dir=transformer_subset_dir,
                                upstream=upstream,
                                permute=True,
                                seed=seed,
                                sample_size=samples_size,
                                verbose=2)

    epochs = int(n_backwards / len(train_dataset))
    print("Runnning {} epochs".format(epochs), flush=True)
    
    best_loss = build_loader_model_trainer(train_dataset=train_dataset,
                                test_datasets=test_datasets,
                                epochs=epochs,
                                dir_out=subset_out,
                                seed=seed,
                                verbose=2)

    print(best_loss, flush=True)
