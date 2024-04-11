import warnings
import pathlib as pl
import pickle

from surrogate.nn.functional import Transformer


def load_transformer(dir: pl.Path,
                             feature: str,
                             verbose: int = 0) -> Transformer:
    transformer_file = pl.Path("{}/{}_transformer.pkl".format(dir, feature))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(transformer_file.resolve(), 'rb') as handle:
            transformer = pickle.load(handle)
            
    if verbose > 0:
        print("Loaded transformer")
    return transformer
