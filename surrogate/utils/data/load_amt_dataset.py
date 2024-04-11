from typing import Optional
import pathlib as pl

import numpy as np

from surrogate.nn.functional import Transformer

from .load_amt import load_amt
from .concatenate_arrays_metas import concatenate_arrays
from .concatenate_arrays_metas import concatenate_metas
from .sort_array_meta import sort_array_meta
from .split_array_meta import split_array_meta


def load_amt_dataset(feature: str,
                     array_dir: pl.Path,
                     meta_dir: Optional[pl.Path] = None,
                     transformer_dir: Optional[pl.Path] = None,
                     transformer: Optional[Transformer] = None,
                     split_fraction: float = 1.0,
                     seed: int = 19920223,
                     verbose: int = 1) -> tuple[np.ndarray, Optional[dict], Optional[Transformer]]:
    
    domains = ["."] + [dir.stem for dir in array_dir.iterdir() if dir.is_dir()]
    
    arrays = []
    metas = []
    for domain in domains:
        if verbose > 2:
            print("Processing {} domain".format(domain))
            
        array_domain_dir = pl.Path("{}/{}".format(array_dir, domain))
        
        meta_domain_dir = None
        if meta_dir is not None:
            meta_domain_dir = pl.Path("{}/{}".format(meta_dir, domain))
        
        array_file = pl.Path("{}/{}.npy".format(array_domain_dir, feature))
        if not array_file.exists():
            continue
        
        array, meta, transformer = load_amt(feature=feature,
                                            array_dir=array_domain_dir,
                                            meta_dir=meta_domain_dir,
                                            transformer_dir=transformer_dir,
                                            transformer=transformer,
                                            verbose = verbose - 2)
        
        if transformer is not None and transformer_dir is not None:
            transformer_dir = None # Do not use transformer_dir next loop
        
        if array.size <= 0:
            continue # no samples
        
        arrays.append(array)
        metas.append(meta)
    
    if len(arrays) == 0:
        raise ValueError("Feature {} not found in {} (with domains {})".format(feature,
                                                                               array_dir,
                                                                               domains))
        
    array = concatenate_arrays(arrays=arrays,
                                direction="sample")
    
    meta = None
    if len(metas) > 0:
        meta = concatenate_metas(metas = metas,
                                 direction = "sample")
    
    if meta is not None:
        array, meta = sort_array_meta(array = array,
                                      meta = meta,
                                      verbose = verbose - 1)
    else:
        raise Warning("Note that, without using a meta, the array samples cannot be sorted")
    
    if split_fraction < 1.0:
        array, meta = split_array_meta(array = array,
                                        meta = meta,
                                        split_fraction=split_fraction,
                                        seed=seed,
                                        verbose=verbose-1)
    
    if verbose > 0:
        print("Loaded {} domain arrays: {} with metas and transformers".format(len(arrays),
                                                                                array.shape))

    return array, meta, transformer
