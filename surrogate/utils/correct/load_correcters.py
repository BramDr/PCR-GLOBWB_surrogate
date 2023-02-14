import pathlib as pl
import pickle

from .Correcter import Correcter

def load_correcter(file: pl.Path,
              verbose: int = 0) -> Correcter:
    with open(file, 'rb') as handle:
        correcter = pickle.load(handle)

    if verbose > 0:
        print("Loaded correcter", flush=True)

    return correcter


def load_correcters(files: list[pl.Path],
               verbose: int = 1) -> list[Correcter]:

    correcters = []
    for meta_file in files:
        correcter = load_correcter(file=meta_file,
                         verbose=verbose - 1)
        correcters.append(correcter)

    if verbose > 0:
        print("Loaded {} correcters".format(len(correcters)), flush=True)

    return correcters
