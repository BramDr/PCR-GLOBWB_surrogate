import pathlib as pl
import pickle

import numpy as np
import torch

from surrogate.utils.data import load_batchset_sequence
from surrogate.nn import SurrogateModel

setup_dir = pl.Path("../../setup/landsurface/saves/train-test2")
train_dir = pl.Path("./saves/train-test2")
out_dir = pl.Path("./saves/train-test2")

samples_size = 32
trainset = "test"
dataset = "output"

mresolutions = [dir.stem for dir in train_dir.iterdir() if dir.is_dir()]
 
mresolution = mresolutions[0]
for mresolution in mresolutions:
    print("Model resolution: {}".format(mresolution))
    
    train_resolution_dir = pl.Path("{}/{}".format(train_dir, mresolution))
    out_mresolution_dir = pl.Path("{}/{}".format(out_dir, mresolution))
        
    model = SurrogateModel.load(save_dir=train_resolution_dir)
    print(model)
    
    if model.input_features is None or model.input_transformers is None:
        raise ValueError("Model has no input features or transformers.")
    if model.output_features is None or model.output_transformers is None:
        raise ValueError("Model has no output features or transformers.")
    
    dresolutions = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]

    dresolution = dresolutions[0]
    for dresolution in dresolutions:       
        # if mresolution != "mulres" and mresolution != dresolution:
        #     continue
            
        print("\t\tData resolution: {}".format(dresolution))
        
        setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, dresolution))
        out_dresolution_dir = pl.Path("{}/{}".format(out_mresolution_dir, dresolution))
        
        setup_trainset_dir = pl.Path("{}/{}".format(setup_resolution_dir, trainset))
        out_trainset_dir = pl.Path("{}/{}".format(out_dresolution_dir, trainset))
        
        out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))
        
        exists = True
        for feature_index, feature in enumerate(model.output_features):
            array_file = pl.Path("{}/{}.npy".format(out_dataset_dir, feature))
            meta_file = pl.Path("{}/{}_meta.pkl".format(out_dataset_dir, feature))
            if not array_file.exists() or not  meta_file.exists():
                exists = False
                break
        if exists:
            print("Already done")
            continue
                    
        test_batchset = load_batchset_sequence(array_dir=setup_trainset_dir,
                                                meta_dir=setup_trainset_dir,
                                                input_features=model.input_features,
                                                output_features=model.output_features,
                                                input_transformers=model.input_transformers,
                                                output_transformers=model.output_transformers,
                                                samples_size=samples_size,
                                                include_output=False,
                                                cuda=False,
                                                verbose=1)

        y_preds = []
        for sample_index in range(test_batchset.mapped_samples_len):
            hidden = None
            cell = None
            
            y_preds_dates = []
            for date_index in range(test_batchset.mapped_dates_len):
                x = test_batchset.x_mapped[sample_index][date_index]
                
                with torch.inference_mode():
                    y_pred, (hidden, cell) = model.forward(x.cuda(), hidden, cell)
                    y_pred = y_pred.detach().cpu()
                y_preds_dates.append(y_pred)
            y_pred = np.concatenate(y_preds_dates, axis=0)
            y_preds.append(y_pred)
        y_pred = np.concatenate(y_preds, axis=-2)
        del y_preds
        
        feature_index = 0
        feature = model.output_features[feature_index]
        for feature_index, feature in enumerate(model.output_features):
            print("\t\tFeature: {}".format(feature))

            array_file = pl.Path("{}/{}.npy".format(out_dataset_dir, feature))
            meta_file = pl.Path("{}/{}_meta.pkl".format(out_dataset_dir, feature))
            if array_file.exists() and meta_file.exists():
                print("Already done")
                continue
        
            array = y_pred[..., [feature_index]]
        
            transformer = model.output_transformers[feature]
            if transformer is None:
                raise ValueError("Model has no transformer for feature {}.".format(feature))
            array = transformer.detransform(array)
            
            array_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_file, array)
            
            meta = {"samples": test_batchset.samples,
                    "lats": test_batchset.lats,
                    "lons": test_batchset.lons,
                    "dates": test_batchset.dates,
                    "spatial_mapping": np.arange(test_batchset.samples.size),
                    "dates_mapping": np.arange(test_batchset.dates.size),}
            
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_file, "wb") as handle:
                pickle.dump(meta, handle)
            