from typing import Optional, Sequence
import pathlib as pl

import numpy as np

from .load_metas import load_meta
from .load_arrays import load_array
from .load_transformers import load_transformer
from .concatenate_arrays_metas import concatenate_arrays_metas


def load_concatenate_array_meta(save_dir: pl.Path,
                                dataset: str,
                                features: Optional[Sequence[str]] = None,
                                transformer_dir: Optional[pl.Path] = None,
                                seperate_files: bool = True,
                                verbose: int = 1) -> tuple[np.ndarray, dict]:

    subset_dir = pl.Path("{}/{}".format(save_dir, dataset))
    
    array_files = [file for file in subset_dir.rglob("*.npy")]
    if seperate_files and features is not None:
        array_files = [file for feature in features for file in subset_dir.rglob("{}.npy".format(feature))]
    
    metas = []
    arrays = []
    for array_file in array_files:
        meta_file = pl.Path("{}/{}_meta.pkl".format(array_file.parent, array_file.stem))
        meta = load_meta(file=meta_file,
                        verbose=verbose - 1)

        transformer = None
        if transformer_dir is not None:            
            transformer_file = pl.Path("{}/{}/{}_transformer.pkl".format(transformer_dir, array_file.parent.stem, array_file.stem))
            transformer = load_transformer(file=transformer_file, verbose=verbose - 1)

        array = load_array(file=array_file,
                           meta=meta,
                           transformer=transformer,
                           verbose=verbose - 1)
        
        meta["dataset"] = dataset
        if seperate_files:
            meta["features"] = [array_file.stem]
        elif features is not None:
            features_sel =  [feature in features for feature in meta["features"]]
            array = array[..., features_sel]
            meta["features"] = [feature for feature in meta["features"] if feature in features]
        
        metas.append(meta)
        arrays.append(array)

    array, meta = concatenate_arrays_metas(arrays=arrays,
                                           metas=metas,
                                           direction="feature",
                                           verbose=verbose)

    return array, meta


def load_concatenate_arrays_metas(save_dirs: Sequence[pl.Path],
                                  dataset: str,
                                  features: Optional[Sequence[str]] = None,
                                  transformer_dir: Optional[pl.Path] = None,
                                  seperate_files: bool = True,
                                  verbose: int = 1) -> tuple[list[np.ndarray], list[dict]]:
    arrays = []
    metas = []
    for save_dir in save_dirs:
        array, meta = load_concatenate_array_meta(save_dir=save_dir,
                                                  dataset=dataset,
                                                  features=features,
                                                  transformer_dir=transformer_dir,
                                                  seperate_files=seperate_files,
                                                  verbose=verbose)

        arrays.append(array)
        metas.append(meta)

    return arrays, metas
