from typing import Optional
import numpy as np
import numba as nb

from surrogate.nn.functional import Transformer


def _fill_missing(array: np.ndarray,
                  missing_value: float = 0,
                  verbose: int = 1) -> np.ndarray:
    
    missing = np.isnan(array)
    array = np.where(missing, missing_value, array)
    
    if verbose > 0:
        print("Filled array {}".format(array.shape))
    return array

def _transform_array(array: np.ndarray,
                     transformer: Transformer,
                     verbose: int = 1) -> np.ndarray:
    
    array = transformer.transform(array)
    
    if verbose > 0:
        print("Transformed array {}".format(array.shape))
    return array

@nb.njit
def _map_spatial_indices(array: np.ndarray,
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


@nb.njit
def _map_temporal_indices(array: np.ndarray,
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


def _expand_array_samples(array: np.ndarray,
                          meta: dict) -> np.ndarray:

    samples_len = len(meta["samples"])
    expanded_shape = list(array.shape)
    expanded_shape[-2] = samples_len
    expanded_shape = tuple(expanded_shape)
    
    if array.shape[-2] == 1:
        array = np.broadcast_to(array=array,
                                shape=expanded_shape)
    elif array.shape[-2] != samples_len:
        array = _map_spatial_indices(array=array,
                                    mapping=meta["spatial_mapping"],
                                    mapped_shape=expanded_shape)
    return array


def _expand_array_dates(array: np.ndarray,
                        meta: dict) -> np.ndarray:

    dates_len = len(meta["dates"])
    expanded_shape = list(array.shape)
    expanded_shape[0] = dates_len
    expanded_shape = tuple(expanded_shape)

    if array.shape[0] == 1:
        array = np.broadcast_to(array=array,
                                shape=expanded_shape)
    elif array.shape[0] != dates_len:
        array = _map_temporal_indices(array=array,
                                        mapping=meta["date_mapping"],
                                        mapped_shape=expanded_shape)
    return array


def _expand_array(array: np.ndarray,
                  meta: dict,
                  verbose: int = 1) -> np.ndarray:

    array = _expand_array_samples(array=array,
                                  meta=meta)
    array = _expand_array_dates(array=array,
                                meta=meta)
    if verbose > 0:
        print("Expanded array {}".format(array.shape))
    return array


def modify_array(array: np.ndarray,
                 fill_missing: bool = True,
                 meta: Optional[dict] = None,
                 transformer: Optional[Transformer] = None,
                 verbose: int = 1) -> np.ndarray:

    array = array.astype(np.float32)
    
    while len(array.shape) < 2:
        array = np.expand_dims(a=array, axis=0)
    
    while len(array.shape) < 3:
        array = np.expand_dims(a=array, axis=-1)

    if array.size > 0:
        if fill_missing:
            array = _fill_missing(array=array,
                                  verbose=verbose)

        if transformer is not None:
            array = _transform_array(array=array,
                                     transformer=transformer,
                                     verbose=verbose)

        if meta is not None:
            array = _expand_array(array=array,
                                  meta=meta,
                                  verbose=verbose)

    if verbose > 0:
        print("Modified array {}".format(array.shape))
    return array
