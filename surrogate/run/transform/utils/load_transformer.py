import pathlib as pl
import warnings

from surrogate.nn.functional import Transformer
from surrogate.utils.data import load_array


def load_transformer(array_file: pl.Path,
                      transformer: Transformer,
                      verbose: int = 1) -> Transformer:
    
    array = load_array(file=array_file,
                       verbose=verbose-1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformer.fit(input = array)
    
    if verbose > 0:
        print("Fitted {}".format(transformer))

    return transformer
