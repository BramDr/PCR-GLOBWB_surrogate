from __future__ import annotations
from typing import Union
import warnings

import torch
import numpy as np

from .Transformer import Transformer

class MinMaxTransformer(Transformer):
    def __init__(self) -> None:
        super(MinMaxTransformer, self).__init__()
        
        self.min = None
        self.max = None
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> MinMaxTransformer:
        
        input = np.array(input)
        
        axis = [dim for dim in range(input.ndim - 1)]
        axis = tuple(axis)
        
        self.min = np.min(input, axis=axis, keepdims=True)
        self.max = np.max(input, axis=axis, keepdims=True)
        self.valid = np.copy(self.max != self.min)
        
        self.state = {"min": self.min,
                      "max": self.max,
                      "valid": self.valid}
        
        return self

    def _transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.min is None or self.max is None or self.valid is None:
            raise ValueError("Transformer not fitted")
        
        if isinstance(input, torch.Tensor):
            min = torch.from_numpy(self.min).to(input.device)
            max = torch.from_numpy(self.max).to(input.device)
            valid = torch.from_numpy(self.valid).to(input.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output = torch.where(valid, (input - min) / (max - min), 0)
        elif isinstance(input, np.ndarray):
            min = np.array(self.min)
            max = np.array(self.max)
            valid = np.array(self.valid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output = np.where(valid, (input - min) / (max - min), 0)
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.min is None or self.max is None or self.valid is None:
            raise ValueError("Transformer not fitted")
        
        min = self.min
        max = self.max
        if isinstance(input, torch.Tensor):
            min = torch.Tensor(self.min).to(input.device)
            max = torch.Tensor(self.max).to(input.device)
            output = input * (max - min) + min
        elif isinstance(input, np.ndarray):
            min = np.array(self.min)
            max = np.array(self.max)
            output = input * (max - min) + min
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return input
    