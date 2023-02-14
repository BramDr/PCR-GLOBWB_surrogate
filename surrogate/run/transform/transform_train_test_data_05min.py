import pathlib as pl
import pickle
import re

import sklearn.preprocessing as pp

from surrogate.nn.functional import SklearnTransformer

from utils.load_transformer import load_transformer

prepare_dir = pl.Path("../prepare/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
seed = 19920223

submasks = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    prepare_submask_dir = pl.Path("{}/{}".format(prepare_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))
    
    trainsets = [dir.stem for dir in prepare_submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

    trainset = trainsets[0]
    for trainset in trainsets:
        print("Working on {}".format(trainset), flush=True)

        prepare_subset_dir = pl.Path("{}/{}/cells_training".format(prepare_submask_dir, trainset))
        trainset_out = pl.Path("{}/{}".format(submask_out, trainset))

        subset_files = [file for file in prepare_subset_dir.rglob("*.npy")]

        subset_file = subset_files[0]
        for subset_file in subset_files:
            print("Working on {}".format(subset_file.stem), flush=True)
            
            quantile_transformer = pp.QuantileTransformer(n_quantiles=10000)
            transformer = SklearnTransformer(transformer=quantile_transformer)
            
            transformer = load_transformer(array_file=subset_file,
                                            transformer=transformer,
                                            verbose=1)
            
            transformer_out = pl.Path("{}/{}/{}_transformer.pkl".format(trainset_out, subset_file.parent.stem, subset_file.stem))    
            transformer_out.parent.mkdir(parents=True, exist_ok=True)
            with open(transformer_out, 'wb') as file:
                    pickle.dump(transformer, file)
