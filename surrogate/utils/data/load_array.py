from typing import Optional, Sequence
import pathlib as pl
import numpy as np
import numba as nb

from surrogate.nn.functional import Transformer

from .modify_array import modify_array


def load_array(dir: pl.Path,
               feature: str,
               meta: Optional[dict] = None,
               transformer: Optional[Transformer] = None,
               verbose: int = 0) -> np.ndarray:
    
    array_file = pl.Path("{}/{}.npy".format(dir, feature))
    array = np.load(array_file.resolve())
    
    array = modify_array(array=array,
                         meta=meta,
                         transformer=transformer,
                         verbose=verbose - 1)
    
    if verbose > 0:
        print("Loaded array {}".format(array.shape))
    return array
