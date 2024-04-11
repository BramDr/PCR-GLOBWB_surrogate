from typing import Optional, Sequence, Union
import pathlib as pl

import numpy as np

from surrogate.nn.functional import Transformer

from .load_amt_dataset import load_amt_dataset
from .concatenate_arrays_metas import concatenate_metas


def load_amt_dataset_features(features: Union[Sequence, np.ndarray],
                              array_dir: pl.Path,
                              meta_dir: Optional[pl.Path] = None,
                              transformer_dir: Optional[pl.Path] = None,
                              transformers: Optional[dict[str, Optional[Transformer]]] = None,
                              split_fraction: float = 1.0,
                              seed: int = 19920223,
                              verbose: int = 1) -> tuple[np.ndarray, Optional[dict], Optional[dict[str, Optional[Transformer]]]]:
    
    if transformers is None:
        transformers = dict(zip(features, [None for _ in features]))
    else:
        check_features = np.array(list(transformers.keys()))
        check_sel = np.isin(check_features, features)
        if check_sel.sum() != check_sel.size:
            raise ValueError("Transformer feature(s) {} are not known".format(check_features[check_sel]))
    
    check_features = np.array(features)
    check_sel = np.isin(features, list(transformers.keys()))
    if check_sel.sum() != check_sel.size:
        raise ValueError("Not all feature(s) {} in Transformers".format(check_features[check_sel]))
    
    features_array = None
    metas = []
    for feature_index, feature in enumerate(features):
        if verbose > 1:
            print("Processing {}".format(feature))
        
        transformer = transformers[feature]
            
        array, meta, transformer = load_amt_dataset(feature=feature,
                                                    array_dir=array_dir,
                                                    meta_dir=meta_dir,
                                                    transformer_dir=transformer_dir,
                                                    transformer=transformer,
                                                    split_fraction=split_fraction,
                                                    seed=seed,
                                                    verbose=verbose - 2)
        
        # Allocate array for all features
        if features_array is None:
            shape = (array.shape[0], array.shape[1], len(features))
            features_array = np.empty(shape=shape, dtype=array.dtype)
            
        features_array[..., feature_index] = array[..., 0]
        
        metas.append(meta)
        transformers[feature] = transformer
    
    if features_array is None:
        raise ValueError("Features {} not found in {}".format(features,
                                                              array_dir))
        
    meta = None
    if len(metas) > 0:
        meta = concatenate_metas(metas = metas,
                                 direction = "feature")

    if verbose > 0:
        print("Loaded {} feature arrays: {} with metas and transformers".format(len(features),
                                                                                features_array.shape))
    return features_array, meta, transformers