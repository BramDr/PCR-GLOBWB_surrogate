import pathlib as pl
import pickle


def load_meta(file: pl.Path,
              verbose: int = 0) -> dict:
    with open(file, 'rb') as handle:
        meta = pickle.load(handle)

    if verbose > 0:
        print("Loaded meta", flush=True)

    return meta


def load_metas(files: list[pl.Path],
               verbose: int = 1) -> list[dict]:

    metas = []
    for meta_file in files:
        meta = load_meta(file=meta_file,
                         verbose=verbose)
        metas.append(meta)

    return metas
