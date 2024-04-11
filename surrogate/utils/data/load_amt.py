from typing import Optional, Sequence
import pathlib as pl

import numpy as np

from surrogate.nn.functional import Transformer

from .load_meta import load_meta
from .load_transformers import load_transformer
from .load_array import load_array
from .sort_array_meta import sort_array_meta


def load_amt(feature: str,
             array_dir: pl.Path,
             meta_dir: Optional[pl.Path] = None,
             meta: Optional[dict] = None,
             transformer_dir: Optional[pl.Path] = None,
             transformer: Optional[Transformer] = None,
             verbose: int = 1) -> tuple[np.ndarray, Optional[dict], Optional[Transformer]]:
        
    if meta_dir is not None:
        if meta is not None:
            raise ValueError("Cannot use both meta_dir and meta")
        meta = load_meta(dir=meta_dir,
                         feature=feature,
                         verbose=verbose - 1)

    if transformer_dir is not None:
        if transformer is not None:
            raise ValueError("Cannot use both transformer_dir and transformer")
        transformer = load_transformer(dir=transformer_dir,
                                       feature=feature,
                                       verbose=verbose - 1)

    array = load_array(dir=array_dir,
                       feature=feature,
                       meta=meta,
                       transformer=transformer,
                       verbose=verbose - 1)
    
    if meta is not None:
        array, meta = sort_array_meta(array = array,
                                      meta = meta,
                                      verbose = verbose - 1)
    else:
        raise Warning("Note that, without using a meta, the array samples cannot be sorted")
    
    if verbose > 0:
        print("Loaded array: {} with metas and transformers".format(array.shape))
    return array, meta, transformer
