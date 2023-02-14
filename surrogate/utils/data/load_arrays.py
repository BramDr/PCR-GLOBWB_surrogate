from typing import Optional
import pathlib as pl
import numpy as np
import numba as nb

from surrogate.nn.functional import Transformer


def _fill_missing(array: np.ndarray) -> np.ndarray:
    missing = np.isnan(array)

    apply_axis = tuple([axis for axis in range(len(array.shape) - 1)])
    array_mean = np.mean(a=array,
                         axis=apply_axis,
                         keepdims=True,
                         where=~missing)

    array_mean = np.broadcast_to(array=array_mean,
                                 shape=array.shape)

    array = np.where(missing, array_mean, array)

    return array


@nb.njit
def _map_samples_indices(array: np.ndarray,
                         mapping: np.ndarray,
                         mapped_shape: tuple) -> np.ndarray:
    
    mapping_len = len(mapping)
    
    mapped = np.full(mapped_shape,
                     fill_value=np.nan,
                     dtype=np.float32)
    for index in range(mapping_len):
        map_index = mapping[index]
        mapped[index, ...] = array[map_index, ...]

    return mapped


@nb.njit
def _map_dates_indices(array: np.ndarray,
                       mapping: np.ndarray,
                       mapped_shape: tuple) -> np.ndarray:
    
    mapping_len = len(mapping)
    
    mapped = np.full(mapped_shape,
                     fill_value=np.nan,
                     dtype=np.float32)
    for index in range(mapping_len):
        map_index = mapping[index]
        mapped[..., index, :] = array[..., map_index, :]

    return mapped


def _expand_array_samples(array: np.ndarray,
                          meta: dict) -> np.ndarray:

    samples_len = len(meta["samples"])
    expanded_shape = list(array.shape)
    expanded_shape[0] = samples_len
    expanded_shape = tuple(expanded_shape)

    if array.shape[0] == 1:        
        expanded_array = np.broadcast_to(array=array,
                                         shape=expanded_shape)
    else:
        if array.shape[0] == samples_len:
            expanded_array = array
        else:
            expanded_array = _map_samples_indices(array=array,
                                                  mapping=meta["spatial_mapping"],
                                                  mapped_shape=expanded_shape)

    return expanded_array


def _expand_array_dates(array: np.ndarray,
                        meta: dict) -> np.ndarray:

    dates_len = len(meta["dates"])
    expanded_shape = list(array.shape)
    expanded_shape[1] = dates_len
    expanded_shape = tuple(expanded_shape)

    if array.shape[1] == 1:        
        expanded_array = np.broadcast_to(array=array,
                                         shape=expanded_shape)
    else:
        if array.shape[1] == dates_len:
            expanded_array = array
        else:
            expanded_array = _map_dates_indices(array=array,
                                                mapping=meta["dates_mapping"],
                                                mapped_shape=expanded_shape)

    return expanded_array


def _expand_array(array: np.ndarray,
                  meta: dict,
                  verbose: int = 1) -> np.ndarray:

    expanded_array = array
    expanded_array = _expand_array_samples(array = expanded_array,
                                           meta = meta)
    expanded_array = _expand_array_dates(array = expanded_array,
                                           meta = meta)

    if verbose > 0:
        print("Expanded array {}".format(expanded_array.shape), flush=True)

    return expanded_array


def load_array(file: pl.Path,
               meta: Optional[dict] = None,
               transformer: Optional[Transformer] = None,
               detransform: bool = False,
               fill_missing: bool = True,
               verbose: int = 1) -> np.ndarray:

    array = np.load(file).astype(np.float32)

    while len(array.shape) < 3:
        array = np.expand_dims(a=array, axis=-1)

    if verbose > 0:
        print("Loaded array {}".format(array.shape), flush=True)

    if array.size > 0:
        if fill_missing:
            array = _fill_missing(array=array)

        if transformer is not None:
            if detransform:
                array = transformer.detransform(array)
            else:
                array = transformer.transform(array)
                
            if verbose > 0:
                print("Transformed array {}".format(array.shape), flush=True)

    if meta is not None:
        array = _expand_array(array=array,
                                       meta=meta,
                                       verbose=verbose)

    return array


def load_arrays(array_files: list[pl.Path],
                metas: Optional[list[Optional[dict]]] = None,
                transformers: Optional[list[Optional[Transformer]]] = None,
                verbose: int = 1) -> list[np.ndarray]:

    if metas is None:
        metas = [None for _ in range(len(array_files))]
    if transformers is None:
        transformers = [None for _ in range(len(array_files))]

    arrays = []
    for array_file, meta, transformer in zip(array_files, metas, transformers):
        array = load_array(file=array_file,
                           meta=meta,
                           transformer=transformer,
                           verbose=verbose)
        arrays.append(array)

    return arrays
