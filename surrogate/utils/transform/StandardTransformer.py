from __future__ import annotations
from typing import Union
import warnings

import torch
import numpy as np

from .Transformer import Transformer

class StandardTransformer(Transformer):
    def __init__(self) -> None:
        super(StandardTransformer, self).__init__()
        
        self.mean = None
        self.std = None
    
    def fit(self,
            array: Union[np.ndarray, torch.Tensor]) -> StandardTransformer:
        
        array = np.array(array)
        self.mean = np.mean(array)
        self.std = np.std(array)
        self.state = {"mean": self.mean,
                      "std": self.std}
        return self

    def _transform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.std > 0:
                output = (array - self.mean) / self.std
            else:
                output = array * 0
        return output

    def _detransform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:        
        output = array * self.std + self.mean
        return output
    