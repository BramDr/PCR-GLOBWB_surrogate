import pathlib as pl
import warnings

import numpy as np

from surrogate.nn.functional import Transformer
from surrogate.utils.data import load_array


def generate_transformer_array(array: np.ndarray,
                               transformer: Transformer,
                               verbose: int = 1) -> Transformer:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformer.fit(input=array)

    if verbose > 0:
        print("Fitted {}".format(transformer))

    return transformer


def generate_transformer(array_file: pl.Path,
                         transformer: Transformer,
                         verbose: int = 1) -> Transformer:

    array = load_array(file=array_file,
                       verbose=verbose-1)

    transformer = generate_transformer_array(array=array,
                                             transformer=transformer,
                                             verbose=verbose)

    return transformer
