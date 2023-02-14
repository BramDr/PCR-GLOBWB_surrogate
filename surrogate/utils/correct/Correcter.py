from typing import Optional
import warnings

import numpy as np
import numba as nb


@nb.njit
def _get_indices(from_array: np.ndarray,
                 to_array: np.ndarray) -> np.ndarray:
    
    indices = np.full(shape=to_array.shape,
                      fill_value=np.nan,
                      dtype=np.int64)
        
    indices = np.searchsorted(a = from_array, v = to_array, side="left")
            
    max_index = from_array.shape[0] - 1
    indices = np.where(indices >= max_index, max_index, indices)
    
    return indices


def _get_fraction(from_array: np.ndarray,
                  to_array: np.ndarray,
                  indices: np.ndarray) -> np.ndarray:
    
    prev_indices = indices - 1
    prev_indices = np.where(prev_indices < 0, 0, prev_indices)
    
    fractions = np.full(shape=to_array.shape,
                        fill_value=np.nan,
                        dtype=np.float32)
            
    min = _get_indices_values(array=from_array,
                              indices=prev_indices)
    max = _get_indices_values(array=from_array,
                              indices=indices)
    diff = np.subtract(max, min)

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore",
                              category=RuntimeWarning)
        fractions = np.subtract(to_array, min) / diff
    fractions = np.where(diff == 0, 0, fractions)
        
    return fractions


@nb.njit
def _get_indices_values(array: np.ndarray,
                        indices: np.ndarray,
                        fractions: Optional[np.ndarray] = None) -> np.ndarray:
    
    values = array[indices]
    
    if fractions is not None:
        prev_indices = indices - 1
        prev_indices = np.where(prev_indices < 0, 0, prev_indices)     
        prev_values = array[prev_indices]
        
        diff = np.subtract(prev_values, values)
        values = values + diff * fractions

    return values


class Correcter():
    def __init__(self,
                 true: np.ndarray,
                 pred: np.ndarray,
                 max_values: Optional[int] = None):
        true = np.sort(true.flatten())
        pred = np.sort(pred.flatten())

        if max_values is not None:
            agg_factor = int(true.shape[0] / max_values)
            true = true[0:-1:agg_factor, :]
            pred = pred[0:-1:agg_factor, :]

        self.true = true
        self.pred = pred

    def correct(self,
                array: np.ndarray,
                interpolate: bool = True) -> tuple[np.ndarray, tuple[np.ndarray, Optional[np.ndarray]]]:
        
        array_shape = array.shape
        array = array.flatten()

        indices = _get_indices(from_array=self.pred,
                               to_array=array)

        fractions = None
        if interpolate:
            fractions = _get_fraction(from_array=self.pred,
                                      to_array=array,
                                      indices=indices)
            
        corrected = _get_indices_values(array=self.true,
                                        indices=indices,
                                        fractions=fractions)
        
        corrected = np.reshape(corrected, array_shape)
        
        return corrected, (indices, fractions)
