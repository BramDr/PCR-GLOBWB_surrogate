import warnings as wr
import pathlib as pl
import pickle

from surrogate.nn.functional import Transformer


def load_transformer(file: pl.Path,
                     verbose: int = 0) -> Transformer:

    with wr.catch_warnings():
        wr.simplefilter("ignore")
        with open(file, 'rb') as handle:
            transformer = pickle.load(handle)

    if verbose > 0:
        print("Loaded transformer", flush=True)

    return transformer


def load_transformers(files: list[pl.Path],
                      verbose: int = 1) -> list[Transformer]:

    transformers = []
    for transformer_file in files:
        transformer = load_transformer(file=transformer_file,
                                       verbose=verbose)
        transformers.append(transformer)

    return transformers
