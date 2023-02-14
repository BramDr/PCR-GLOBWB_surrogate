import pathlib as pl

import sklearn.preprocessing as pp

from surrogate.nn.functional import SklearnTransformer
from utils.store_transformers import store_transformers

prepare_dir = pl.Path("../prepare/saves/global_30min")
dir_out = pl.Path("./saves/global_30min")
seed = 19920223

subsets = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

subset = "hyper"
for subset in subsets:
    print("Working on {}".format(subset), flush=True)

    prepare_subset_dir = pl.Path("{}/{}/cells_training".format(prepare_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))

    subset_files = [file for file in prepare_subset_dir.rglob("*.npy")]

    quantile_transformer = pp.QuantileTransformer(n_quantiles=10000)
    transformer = SklearnTransformer(transformer=quantile_transformer)

    store_transformers(transformer=transformer,
                       subset_files=subset_files,
                       dir_out=subset_out,
                       verbose=1)
