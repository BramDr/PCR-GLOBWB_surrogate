import pathlib as pl

import numpy as np
import pandas as pd
import torch

from surrogate.utils.data import load_batchsets
from surrogate.utils.data import combine_spatiotemporal_datasets
from surrogate.nn import load_model
from surrogate.utils.correct import load_correcter
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_05min")
feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
transform_dir = pl.Path("../transform/saves/global_05min")
train_dir = pl.Path("../train/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
seed = 19920232
samples_size = 32
dates_size = 365
trainset = "train_32"

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    submask_dir = pl.Path("{}/{}".format(save_dir, submask))
    prepare_submask_dir = pl.Path("{}/{}".format(prepare_dir, submask))
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    train_submask_dir = pl.Path("{}/{}".format(train_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))

    subsets = [dir.stem for dir in submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

    trainset_dir = pl.Path("{}/total".format(submask_dir))
    prepare_trainset_dir = pl.Path("{}/{}".format(prepare_submask_dir, trainset))
    transform_trainset_dir = pl.Path("{}/{}".format(transform_submask_dir, trainset))
    train_trainset_dir = pl.Path("{}/total".format(train_submask_dir))
    trainset_out = pl.Path("{}/total".format(submask_out))

    corrected_file = pl.Path("{}/corrected.npy".format(trainset_out))
    corrected_file.parent.mkdir(parents=True, exist_ok=True)
    ts_file = pl.Path("{}/output_ts.csv".format(trainset_out))
    ts_file.parent.mkdir(parents=True, exist_ok=True)
    cdf_file = pl.Path("{}/output_cdf.csv".format(trainset_out))
    cdf_file.parent.mkdir(parents=True, exist_ok=True)
    sp_file = pl.Path("{}/output_sp.csv".format(trainset_out))
    sp_file.parent.mkdir(parents=True, exist_ok=True)

    if ts_file.exists() and cdf_file.exists() and sp_file.exists():
        continue

    subset_dirs = [pl.Path("{}/cells_training".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_spatial".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_temporal".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_spatiotemporal".format(prepare_trainset_dir))]

    datasets = load_batchsets(save_dirs=subset_dirs,
                                input_features=in_features["feature"].to_list(),
                                output_features=out_features["feature"].to_list(),
                                transformer_dir=transform_trainset_dir,
                                include_output=False,
                                sample_size=samples_size,
                                #dates_size=dates_size,
                                verbose=2)

    dataset = combine_spatiotemporal_datasets(train=datasets[0],
                                              spatial_test=datasets[1],
                                              temporal_test=datasets[2],
                                              spatiotemporal_test=datasets[3],
                                              verbose=2)
    del datasets

    state_file = pl.Path("{}/state_dict.pt".format(train_trainset_dir))
    model = load_model(state_file=state_file,
                        dropout_rate=0,
                        try_cuda=True,
                        verbose=1)

    y_preds = []
    for index in range(len(dataset)):
        x, y_true = dataset[index]
        with torch.inference_mode():
            y_pred, _ = model.forward(x.cuda())
            y_pred = y_pred.detach().cpu()        
        y_preds.append(y_pred)
    y_pred = torch.concat(y_preds, dim = 0).numpy()
    del y_preds

    y_corrs = []
    index = 0
    feature = out_features["feature"].iloc[index]
    for index, feature in enumerate(out_features["feature"]):
        
        corrector_file = pl.Path("{}/{}_corrector.pkl".format(trainset_out, feature))
        corrector = load_correcter(file=corrector_file)
        
        feature_pred = y_pred[..., index]
        feature_corr, (indices, fractions) = corrector.correct(feature_pred)
        y_corrs.append(feature_corr)        
    y_corr = np.stack(y_corrs, axis = -1)
    del y_corrs
    del y_pred
    
    np.save(corrected_file, y_corr)

    plot_df = array_to_ts_dataframe(input=y_corr,
                                    features=out_features["feature"].to_list(),
                                    sequences=dataset.dates,
                                    save_sensitivity=True,
                                    save_range=True)
    plot_df.to_csv(ts_file)

    plot_df = array_to_cdf_dataframe(input=y_corr,
                                    features=out_features["feature"].to_list(),
                                    max_values=1000)
    plot_df.to_csv(cdf_file)

    if dataset.lats is None or dataset.lons is None:
        raise ValueError("No lats or lons found in dataset")

    plot_df = array_to_sp_dataframe(input=y_corr,
                                    features=out_features["feature"].to_list(),
                                    lats=dataset.lats,
                                    lons=dataset.lons,
                                    save_variability=True,
                                    save_range=True)
    plot_df.to_csv(sp_file)
