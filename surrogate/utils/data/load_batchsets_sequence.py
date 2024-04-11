from typing import Optional, Sequence

import pathlib as pl
import numpy as np

from surrogate.nn.functional import Transformer

from .RecurrentBatchsetSequence import RecurrentBatchsetSequence
from .load_amt_dataset_features import load_amt_dataset_features


def load_batchset_sequence(array_dir: pl.Path,
                           meta_dir: pl.Path,
                           input_subset: str = "input",
                           output_subset: str = "output",
                           input_features: Optional[np.ndarray] = None,
                           output_features: Optional[np.ndarray] = None,
                           transformer_dir: Optional[pl.Path] = None,
                           input_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                           output_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                           include_input: bool = True,
                           include_output: bool = True,
                           samples_size: Optional[int] = None,
                           dates_size: Optional[int] = None,
                           split_fraction: float = 1.0,
                           cuda: bool = False,
                           seed: int = 19920223,
                           verbose: int = 1) -> RecurrentBatchsetSequence:

    input_array = None
    input_meta = None
    if include_input:
        if input_features is None:
            raise ValueError("Input features are None")
        
        input_array_dir = pl.Path("{}/{}".format(array_dir, input_subset))
        input_meta_dir = pl.Path("{}/{}".format(meta_dir, input_subset))
        input_transformer_dir = None
        if transformer_dir is not None:
            input_transformer_dir = pl.Path("{}/{}".format(transformer_dir, input_subset))
        
        input_array, input_meta, input_transformers = load_amt_dataset_features(features=input_features,
                                                                                array_dir=input_array_dir,
                                                                                meta_dir=input_meta_dir,
                                                                                transformer_dir=input_transformer_dir,
                                                                                transformers=input_transformers,
                                                                                split_fraction=split_fraction,
                                                                                seed=seed,
                                                                                verbose=verbose-1)
        
        if input_meta is None:
            raise ValueError("Metas are None")

    output_array = None
    output_meta = None
    if include_output:
        if output_features is None:
            raise ValueError("Output features are None")
        
        output_array_dir = pl.Path("{}/{}".format(array_dir, output_subset))
        output_meta_dir = pl.Path("{}/{}".format(meta_dir, output_subset))
        output_transformer_dir = None
        if transformer_dir is not None:
            output_transformer_dir = pl.Path("{}/{}".format(transformer_dir, output_subset))
        
        output_array, output_meta, output_transformers = load_amt_dataset_features(features=output_features,
                                                                                   array_dir=output_array_dir,
                                                                                   meta_dir=output_meta_dir,
                                                                                   transformer_dir=output_transformer_dir,
                                                                                   transformers=output_transformers,
                                                                                   split_fraction=split_fraction,
                                                                                   seed=seed,
                                                                                   verbose=verbose-1)

    x = None
    y = None
    samples = None
    lons = None
    lats = None
    dates = None
    x_features = None
    y_features = None
    x_transformers = None
    y_transformers = None

    if input_array is not None:
        if input_meta is None:
            raise ValueError("Metas are None")
        
        x = input_array
        samples = input_meta["samples"]
        lons = input_meta["lons"]
        lats = input_meta["lats"]
        dates = input_meta["dates"]
        x_features = input_meta["features"]
        x_transformers = input_transformers

    if output_array is not None:
        if output_meta is None:
            raise ValueError("Metas are None")
        
        y = output_array
        samples = output_meta["samples"]
        lons = output_meta["lons"]
        lats = output_meta["lats"]
        dates = output_meta["dates"]
        y_features = output_meta["features"]
        y_transformers = output_transformers

    batchset = RecurrentBatchsetSequence(x=x,
                                         y=y,
                                         samples_size=samples_size,
                                         dates_size=dates_size,
                                         samples=samples,
                                         lons=lons,
                                         lats=lats,
                                         dates=dates,
                                         x_features=x_features,
                                         y_features=y_features,
                                         x_transformers=x_transformers,
                                         y_transformers=y_transformers,
                                         cuda=cuda)

    if verbose > 0:
        print("Loaded {}".format(batchset), flush=True)

    return batchset