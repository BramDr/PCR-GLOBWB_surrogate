import pathlib as pl
import pickle

import pandas as pd
import numpy as np

from surrogate.utils.data import load_amt_dataset
from surrogate.utils.data import concatenate_arrays_metas
from surrogate.utils.data import sort_array_meta

input_dir = pl.Path("./input")
save_dir = pl.Path("./saves/train-test2")
prepare_dir = pl.Path("../../prepare/routing/saves/train-test2/mulres")
out_dir = pl.Path("./saves/train-test2/mulres")
datasets = ["input", "output"]

routing_types = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]

routing_type = routing_types[0]
for routing_type in routing_types:
    print("Routing type: {}".format(routing_type))
    
    prepare_routing_dir = pl.Path("{}/{}".format(prepare_dir, routing_type))
    out_routing_dir = pl.Path("{}/{}".format(out_dir, routing_type))
    
    trainsets = [dir.stem for dir in prepare_routing_dir.iterdir() if dir.is_dir()]

    trainset = trainsets[0]
    for trainset in trainsets:
        print("\tTrainset: {}".format(trainset))
        
        prepare_trainset_dir = pl.Path("{}/{}".format(prepare_routing_dir, trainset))
        out_trainset_dir = pl.Path("{}/{}".format(out_routing_dir, trainset))
        
        cells_file = pl.Path("{}/cells.parquet".format(prepare_trainset_dir))
        cells = pd.read_parquet(cells_file)
        
        resolutions = cells["resolution"].unique()
        
        dataset = datasets[0]
        for dataset in datasets:
            print("\t\tDataset: {}".format(dataset))
            
            out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))
            
            save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolutions[0]))
            save_routing_dir = pl.Path("{}/{}".format(save_resolution_dir, routing_type))
            save_trainset_dir = pl.Path("{}/{}".format(save_routing_dir, trainset))
            save_dataset_dir = pl.Path("{}/{}".format(save_trainset_dir, dataset))
            
            features = np.unique([file.stem for file in save_dataset_dir.rglob("*.npy") if file.is_file()])
            
            feature = features[0]
            for feature in features:
                print("\t\t\tFeature: {}".format(feature))
                
                out_feature_dir = pl.Path("{}/{}".format(out_dataset_dir, feature))
                
                array_file = pl.Path("{}.npy".format(out_feature_dir))
                meta_file = pl.Path("{}_meta.pkl".format(out_feature_dir))
                if array_file.exists() and meta_file.exists():
                    print("Already exists")
                    continue
                
                arrays = []
                metas = []
                
                resolution = resolutions[0]
                for resolution in resolutions:
                    print("\t\t\t\tResolution: {}".format(resolution))

                    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
                    save_routing_dir = pl.Path("{}/{}".format(save_resolution_dir, routing_type))
                    save_trainset_dir = pl.Path("{}/{}".format(save_routing_dir, trainset))
                    save_dataset_dir = pl.Path("{}/{}".format(save_trainset_dir, dataset))
            
                    resolution_cells = cells.loc[cells["resolution"] == resolution]
                    
                    array, meta, _ = load_amt_dataset(feature=feature,
                                                        array_dir=save_dataset_dir,
                                                        meta_dir=save_dataset_dir,
                                                        verbose=0)
                    if meta is None:
                        raise ValueError("Meta is None")
                    
                    samples_sel = np.isin(resolution_cells["resolution_index"], meta["samples"])
                    if samples_sel.sum() != samples_sel.size:
                        raise ValueError("Not all resolution indices in meta samples")
                    
                    samples_sel = np.isin(meta["samples"], resolution_cells["resolution_index"])
                    meta["lons"] = meta["lons"][samples_sel]
                    meta["lats"] = meta["lats"][samples_sel]
                    meta["resolution"] = [resolution] * samples_sel.sum()
                    meta["resolution_samples"] = meta["samples"][samples_sel]
                    array = array[:, samples_sel, :]
                    
                    samples = [resolution_cells[resolution_cells["resolution_index"] == sample].index[0] for sample in meta["resolution_samples"]]
                    meta["samples"] = np.array(samples)
                    
                    arrays.append(array)
                    metas.append(meta)
                
                # Concatenate arrays and metas
                array, meta = concatenate_arrays_metas(arrays = arrays,
                                                       metas = metas,
                                                       direction="sample",
                                                       verbose = 0)
                if array is None or meta is None:
                    raise ValueError("Array or meta is None")
                
                meta["resolution"] = np.concatenate([meta["resolution"] for meta in metas])
                meta["resolution_samples"] = np.concatenate([meta["resolution_samples"] for meta in metas])
                
                # Sort array and meta
                sort_indices = np.argsort(meta["samples"])
                array, meta = sort_array_meta(array = array,
                                              meta = meta,
                                              verbose = 0)
                
                meta["resolution"] = meta["resolution"][sort_indices]
                meta["resolution_samples"] = meta["resolution_samples"][sort_indices]
                
                if "origional_lons" in meta.keys():
                    meta.pop("origional_lons")
                if "origional_lats" in meta.keys():
                    meta.pop("origional_lats")
                if "origional_dates" in meta.keys():
                    meta.pop("origional_dates")
                meta["spatial_mapping"] = np.arange(meta["samples"].size)
                meta["date_mapping"] = np.arange(meta["dates"].size)
                
                array_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(array_file, array)
                meta_file.parent.mkdir(parents=True, exist_ok=True)
                with open(meta_file, "wb") as handle:
                    pickle.dump(meta, handle)
                