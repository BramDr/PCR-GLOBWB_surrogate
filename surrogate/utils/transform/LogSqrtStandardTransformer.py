from __future__ import annotations
from functools import partial
from typing import Union

import torch
import numpy as np

from .Transformer import Transformer
from .StandardTransformer import StandardTransformer

class LogSqrtStandardTransformer(Transformer):
    def __init__(self,
                 small: float = 1e-10,
                 add_log: bool = False,
                 log_10: bool = False,
                 add_sqrt: bool = False) -> None:
        super(LogSqrtStandardTransformer, self).__init__()
        
        np_log_fn = np.log
        torch_log_fn = torch.log
        np_pow_fn = np.exp
        torch_pow_fn = torch.exp
        if log_10:
            np_log_fn = np.log10
            torch_log_fn = torch.log10
            np_pow_fn = partial(np.power, 10)
            torch_pow_fn = partial(torch.pow, 10)
            
        self.small = small
        self.total_min = None
        self.subtransformer = StandardTransformer()
        
        self.add_sqrt = add_sqrt
        self.add_log = add_log
        self.np_log_fn = np_log_fn
        self.torch_log_fn = torch_log_fn
        self.np_pow_fn = np_pow_fn
        self.torch_pow_fn = torch_pow_fn
    
    def _subtransform(self,
                      array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        array = array - self.total_min
        array[array < 0] = 0
        if self.add_sqrt:
            array = array**.5
        if self.add_log:
            array = array + self.small
            if isinstance(array, torch.Tensor):
                array = self.torch_log_fn(array)
            elif isinstance(array, np.ndarray):
                array = self.np_log_fn(array)
            else:
                raise TypeError("array must be torch.Tensor or np.ndarray")
        return array
    
    def _subdetransform(self,
                        array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.add_log:
            if isinstance(array, torch.Tensor):
                array = self.torch_pow_fn(array)
            elif isinstance(array, np.ndarray):
                array = self.np_pow_fn(array)
            else:
                raise TypeError("array must be torch.Tensor or np.ndarray")
            array = array - self.small
        if self.add_sqrt:
            array = array**2
        array = array + self.total_min
        return array
    
    def fit(self,
            array: Union[np.ndarray, torch.Tensor]) -> LogSqrtStandardTransformer:
        array = np.array(array)
        self.total_min = np.min(array)
        array = self._subtransform(array=array)
        self.subtransformer = self.subtransformer.fit(array=array)
        self.state = {"subtransformer": self.subtransformer,
                      "total_min": self.total_min}
        
        return self

    def _transform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        array = self._subtransform(array=array)
        array = self.subtransformer.transform(array = array)
        return array

    def _detransform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:        
        array = self.subtransformer.detransform(array = array)
        array = self._subdetransform(array=array)
        return array
    