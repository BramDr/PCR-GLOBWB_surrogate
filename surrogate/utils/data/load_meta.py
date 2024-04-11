import pathlib as pl
import pickle

from .modify_meta import modify_meta


def load_meta(dir: pl.Path,
              feature: str,
              verbose: int = 0) -> dict:
    meta_file = pl.Path("{}/{}_meta.pkl".format(dir, feature))
    with open(meta_file.resolve(), 'rb') as handle:
        meta = pickle.load(handle)
    
    meta = modify_meta(meta=meta,
                       feature=feature,
                       verbose=verbose - 1)
    
    if verbose > 0:
        print("Loaded meta")
    return meta