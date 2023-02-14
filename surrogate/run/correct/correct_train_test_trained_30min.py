import pathlib as pl
import pickle

import numpy as np
import pandas as pd

from surrogate.utils.data import load_batchsets
from surrogate.utils.correct import build_correcter

prepare_dir = pl.Path("../prepare/saves/global_30min")
transform_dir = pl.Path("../transform/saves/global_30min")
train_dir = pl.Path("../train/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min")
seed = 19920232

upstream = pd.read_csv(upstream_file, index_col=0)

# Load output template
subset = "train_96"
prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
train_data_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
train_dataset = load_batchsets(save_dir=train_data_dir,
                            transformer_dir=transform_subset_dir,
                            upstream=upstream,
                            permute=True,
                            seed=seed,
                            sample_size=10,
                            verbose=1)
out_features = train_dataset.y_features

subsets = ["train_{}".format(size)
           for size in [8, 16, 32, 48, 64, 80, 96]]
subsets.reverse()

subset = "train_96"
for subset in subsets:
    print("Working on {}".format(subset), flush = True)

    transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
    train_subset_dir = pl.Path("{}/{}".format(train_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))

    corrector_out = pl.Path("{}/correcter.pkl".format(subset_out))
    corrector_out.parent.mkdir(parents=True, exist_ok=True)
    
    true_df_file = pl.Path("{}/output_cdf.csv".format(transform_subset_dir))
    pred_df_file = pl.Path("{}/output_cdf.csv".format(train_subset_dir))
    
    true_df = pd.read_csv(true_df_file, index_col=0)
    pred_df = pd.read_csv(pred_df_file, index_col=0)
    
    y_trues = []
    y_preds = []
    for feature in out_features:
        true_feature_df = true_df.loc[true_df["feature"] == feature]
        pred_feature_df = pred_df.loc[pred_df["feature"] == feature]
        
        y_true_feature = np.array(true_feature_df["value"])
        y_pred_feature = np.array(pred_feature_df["value"])
        
        y_trues.append(y_true_feature)
        y_preds.append(y_pred_feature)
    
    y_true = np.stack(arrays=y_trues, axis=-1)
    y_pred = np.stack(arrays=y_preds, axis=-1)
    
    corrector = build_correcter(true=y_true,
                                pred=y_pred)

    with open(corrector_out, 'wb') as file:
        pickle.dump(corrector, file)
