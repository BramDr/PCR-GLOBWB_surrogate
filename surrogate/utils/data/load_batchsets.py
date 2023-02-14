from typing import Optional, Sequence

import pathlib as pl
import numpy as np

from .SequenceBatchset import SequenceBatchset
from .load_concatenate_arrays_metas import load_concatenate_array_meta


def load_batchset(save_dir: pl.Path,
                 input_features: Optional[Sequence[str]] = None,
                 output_features: Optional[Sequence[str]] = None,
                 transformer_dir:  Optional[pl.Path] = None,
                 custom_input_arrays: Optional[Sequence[Optional[np.ndarray]]] = None,
                 custom_input_indices: Optional[Sequence[Optional[int]]] = None,
                 include_input: bool = True,
                 include_output: bool = True,
                 sample_size: Optional[int] = None,
                 dates_size: Optional[int] = None,
                 cuda: bool = False,
                 verbose: int = 1) -> SequenceBatchset:
    
    if custom_input_arrays is None:
        custom_input_arrays = [None]
    if custom_input_indices is None:
        custom_input_indices = [None]
    
    input_array = None
    input_meta = None
    if include_input:
        subset = "input"
        input_array, input_meta = load_concatenate_array_meta(save_dir=save_dir,
                                                              dataset=subset,
                                                              features=input_features,
                                                              transformer_dir=transformer_dir,
                                                              verbose=verbose-1)
        
        for array, index in zip(custom_input_arrays, custom_input_indices):
            if array is not None and index is not None:
                array_metric = np.sum(np.mean(array, axis=1))
                input_array_metric = np.sum(np.mean(input_array[:, :, index], axis=1))
                print("Custom input (sum {}) to replace index {} (sum {})".format(array_metric,
                                                                                  index, 
                                                                                  input_array_metric))
                input_array[:, :, [index]] = array
                

    output_array = None
    output_meta = None
    if include_output:
        subset = "output"
        output_array, output_meta = load_concatenate_array_meta(save_dir=save_dir,
                                                                dataset=subset,
                                                                features=output_features,
                                                                transformer_dir=transformer_dir,
                                                                verbose=verbose-1)
    
    x = None
    y = None
    samples = None
    lons = None
    lats = None
    dates = None
    x_features = None
    y_features = None
    
    if input_array is not None and input_meta is not None:
        x = input_array
        samples = input_meta["samples"]
        lons = input_meta["lons"]
        lats = input_meta["lats"]
        dates = input_meta["dates"]
        x_features = input_meta["features"]

    if output_array is not None and output_meta is not None:
        y = output_array
        samples = output_meta["samples"]
        lons = output_meta["lons"]
        lats = output_meta["lats"]
        dates = output_meta["dates"]
        y_features = output_meta["features"]

    dataset = SequenceBatchset(x=x,
                               y=y,
                               samples_size=sample_size,
                               dates_size=dates_size,
                               samples=samples,
                               lons=lons,
                               lats=lats,
                               dates=dates,
                               x_features=x_features,
                               y_features=y_features,
                               cuda=cuda)

    if verbose > 0:            
        print("Loaded {}".format(dataset), flush=True)

    return dataset


def load_batchsets(save_dirs: Sequence[pl.Path],
                   input_features: Optional[Sequence[str]] = None,
                   output_features: Optional[Sequence[str]] = None,
                   transformer_dir:  Optional[pl.Path] = None,
                   include_input: bool = True,
                   include_output: bool = True,
                   sample_size: Optional[int] = None,
                   dates_size: Optional[int] = None,
                   cuda: bool = False,
                   verbose: int = 1) -> Sequence[SequenceBatchset]:

    datasets = []
    for save_dir in save_dirs:
        dataset = load_batchset(save_dir=save_dir,
                               input_features=input_features,
                               output_features=output_features,
                               transformer_dir=transformer_dir,
                               include_input=include_input,
                               include_output=include_output,
                               sample_size=sample_size,
                               dates_size=dates_size,
                               cuda=cuda,
                               verbose=verbose)
        datasets.append(dataset)

    return datasets
