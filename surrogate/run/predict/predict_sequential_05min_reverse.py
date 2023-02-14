import pathlib as pl
import copy
import pickle

import torch
import numpy as np
import pandas as pd

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_array
from surrogate.utils.data import load_transformer
from surrogate.utils.correct import load_correcter
from surrogate.nn import load_model
from surrogate.utils.data import load_batchset
from surrogate.utils.data import concatenate_arrays_metas

from utils.subset_upstream_array import subset_upstream_array

save_dir = pl.Path("./saves/global_05min/reverse")
feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
transform_dir = pl.Path("../transform/saves/global_05min")
correct_dir = pl.Path("../correct/saves/global_05min")
train_dir = pl.Path("../train/saves/global_05min")
dir_out = pl.Path("./saves/global_05min/reverse")
seed = 19920223
samples_size = 32
trainset = "train_32"

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

submasks = [dir.stem for dir in train_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    submask_dir = pl.Path("{}/{}/sequential".format(save_dir, submask))
    feature_submask_dir = pl.Path("{}/{}".format(feature_dir, submask))
    prepare_submask_dir = pl.Path("{}/{}/sequential".format(prepare_dir, submask))
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    correct_submask_dir = pl.Path("{}/{}".format(correct_dir, submask))
    train_submask_dir = pl.Path("{}/{}/total".format(train_dir, submask))
    submask_out = pl.Path("{}/{}/sequential".format(dir_out, submask))

    transform_trainset_dir = pl.Path("{}/{}".format(transform_submask_dir, trainset))
    correct_trainset_dir = pl.Path("{}/total".format(correct_submask_dir))

    discharge_transformer_file = pl.Path("{}/output/discharge_transformer.pkl".format(transform_trainset_dir))
    discharge_transformer = load_transformer(discharge_transformer_file)
    inflow_transformer_file = pl.Path("{}/input/inflow_transformer.pkl".format(transform_trainset_dir))
    inflow_transformer = load_transformer(inflow_transformer_file)

    ldd_file = pl.Path("{}/ldd.csv".format(feature_submask_dir))
    ldd = pd.read_csv(ldd_file, index_col=0)

    state_file = pl.Path("{}/state_dict.pt".format(train_submask_dir))
    model = load_model(state_file=state_file,
                        dropout_rate=0,
                        try_cuda=True,
                        verbose=1)

    sequences = [dir.stem for dir in prepare_submask_dir.iterdir() if dir.is_dir()]
    sequences.sort(key=sort_fn)
    prev_sequences = [None] + sequences[:-2]
    
    prev_sequence = prev_sequences[0]
    sequence = sequences[0]
    for sequence, prev_sequence in zip(sequences, prev_sequences):
        print("Working on {}".format(sequence), flush=True)
        
        prev_sequence_dir = pl.Path("{}/{}".format(submask_dir, prev_sequence))
        prepare_sequence_dir = pl.Path("{}/{}".format(prepare_submask_dir, sequence))
        sequence_out = pl.Path("{}/{}".format(submask_out, sequence))
            
        inflow_file = pl.Path("{}/inflow.npy".format(sequence_out))
        inflow_meta_file = pl.Path("{}/inflow_meta.pkl".format(sequence_out))
        if inflow_file.exists() and inflow_meta_file.exists():
            print("Inflow already present")
            continue
            
        prev_inflow = None
        prev_inflow_meta = None
        if prev_sequence is not None:
            prev_inflow_file = pl.Path("{}/inflow.npy".format(prev_sequence_dir))
            prev_inflow_meta_file = pl.Path("{}/inflow_meta.pkl".format(prev_sequence_dir))
            prev_inflow = np.load(prev_inflow_file)
            prev_inflow_meta = load_meta(file = prev_inflow_meta_file)

        subset_files = [file for file in prepare_sequence_dir.iterdir() if not file.is_dir() and file.suffix == ".csv"]
        
        discharges = []
        discharge_metas = []
        subset_file = subset_files[0]
        for subset_file in subset_files:
            print("Working on {}".format(subset_file.stem), flush=True)

            prepare_subset_dir = pl.Path("{}/{}".format(prepare_sequence_dir, subset_file.stem))
            subset_out = pl.Path("{}/{}".format(sequence_out, subset_file.stem))
            
            y_corr_meta = None
            y_corr = None
            
            predicted_file = pl.Path("{}/predicted.npy".format(subset_out))
            predicted_meta_file = pl.Path("{}/predicted_meta.pkl".format(subset_out))
            if predicted_file.exists() and predicted_meta_file.exists():
                y_corr_meta = load_meta(file = predicted_meta_file)
                y_corr = load_array(file = predicted_file,
                                    meta = y_corr_meta,
                                    verbose=0)
                print("Predicted already present")
            else:
                cells_file = pl.Path("{}.csv".format(prepare_subset_dir))
                cells = pd.read_csv(cells_file, index_col=0)
                subset_samples = cells.index

                subset_inflow = None
                if prev_inflow is not None and prev_inflow_meta is not None:
                    subset_inflow = subset_upstream_array(array = prev_inflow,
                                                          array_samples = prev_inflow_meta["samples"],
                                                          subset_samples = subset_samples,
                                                          ldd=ldd)
                    if subset_inflow is not None:
                        subset_inflow = inflow_transformer.transform(subset_inflow)
                
                inflow_index = in_features["feature"].to_list().index("inflow")
                dataset = load_batchset(save_dir=prepare_subset_dir,
                                        input_features=in_features["feature"].to_list(),
                                        transformer_dir=transform_trainset_dir,
                                        custom_input_arrays=[subset_inflow],
                                        custom_input_indices=[inflow_index],
                                        include_output=False,
                                        sample_size=samples_size,
                                        verbose=2)
                
                # Predict                    
                y_preds = []
                for index in range(len(dataset)):
                    x, _ = dataset[index]
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
                    
                    corrector_file = pl.Path("{}/{}_corrector.pkl".format(correct_trainset_dir, feature))
                    corrector = load_correcter(file=corrector_file)
                    
                    feature_pred = y_pred[..., index]
                    feature_corr, _ = corrector.correct(feature_pred)
                    y_corrs.append(feature_corr)
                y_corr = np.stack(y_corrs, axis = -1)
                del y_corrs
                del y_pred
                
                y_corr_meta = {"samples": dataset.samples,
                               "lons": dataset.lons,
                               "lats": dataset.lats,
                               "dates": dataset.dates,
                               "features": out_features["feature"].to_list()}
                
                predicted_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(predicted_file, y_corr)
                predicted_meta_file.parent.mkdir(parents=True, exist_ok=True)
                with open(predicted_meta_file, "wb") as handler:
                    pickle.dump(y_corr_meta, handler)
            
            # Store dicharge
            discharge_index = out_features["feature"].to_list().index("discharge")
            
            discharge = y_corr[:, :, [discharge_index]]
            discharge_meta = y_corr_meta
            discharge_meta["features"] = [discharge_meta["features"][discharge_index]]
            
            discharge = discharge_transformer.detransform(discharge)
            
            discharges.append(discharge)
            discharge_metas.append(discharge_meta)
        
        inflow, inflow_meta = concatenate_arrays_metas(arrays = discharges,
                                                       metas = discharge_metas,
                                                       direction="sample")
        
        inflow_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(inflow_file, inflow)
        inflow_meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(inflow_meta_file, "wb") as handler:
            pickle.dump(inflow_meta, handler)
