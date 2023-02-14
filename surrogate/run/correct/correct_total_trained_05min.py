import pathlib as pl
import pickle

import numpy as np
import pandas as pd
import torch

from surrogate.utils.data import load_batchsets
from surrogate.utils.data import combine_spatiotemporal_datasets
from surrogate.nn import load_model
from surrogate.utils.correct import build_correcter

feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
transform_dir = pl.Path("../transform/saves/global_05min")
train_dir = pl.Path("../train/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
seed = 19920232
samples_size = 32
trainset = "train_32"

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    prepare_submask_dir = pl.Path("{}/{}".format(prepare_dir, submask))
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    train_submask_dir = pl.Path("{}/{}".format(train_dir, submask))
    submask_out = pl.Path("{}/{}".format(dir_out, submask))

    prepare_trainset_dir = pl.Path("{}/{}".format(prepare_submask_dir, trainset))
    transform_trainset_dir = pl.Path("{}/{}".format(transform_submask_dir, trainset))
    train_trainset_dir = pl.Path("{}/total".format(train_submask_dir))
    trainset_out = pl.Path("{}/total".format(submask_out))
    
    subset_dirs = [pl.Path("{}/cells_training".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_spatial".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_temporal".format(prepare_trainset_dir)),
                   pl.Path("{}/cells_validation_spatiotemporal".format(prepare_trainset_dir))]
    
    datasets = load_batchsets(save_dirs=subset_dirs,
                            input_features=in_features["feature"].to_list(),
                            output_features=out_features["feature"].to_list(),
                            transformer_dir=transform_trainset_dir,
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
    y_trues = []
    for index in range(len(dataset)):
        x, y_true = dataset[index]
        with torch.inference_mode():
            y_pred, _ = model.forward(x.cuda())
            y_pred = y_pred.detach().cpu()
        y_preds.append(y_pred)
        y_trues.append(y_true)

    y_pred = torch.concat(y_preds, dim = 0).numpy()
    y_true = torch.concat(y_trues, dim = 0).numpy()
    
    index = 0
    feature = dataset.y_features[index]
    for index, feature in enumerate(dataset.y_features):
        
        feature_true = y_true[..., index]
        feature_pred = y_pred[..., index]
        corrector = build_correcter(true=feature_true,
                                    pred=feature_pred)

        corrector_out = pl.Path("{}/{}_corrector.pkl".format(trainset_out, feature))
        corrector_out.parent.mkdir(parents=True, exist_ok=True)
        with open(corrector_out, 'wb') as file:
            pickle.dump(corrector, file)
