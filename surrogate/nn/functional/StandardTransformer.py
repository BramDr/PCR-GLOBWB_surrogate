from __future__ import annotations
from typing import Union
import warnings

import torch
import numpy as np

from .Transformer import Transformer
from .MinMaxTransformer import MinMaxTransformer

class StandardTransformer(Transformer):
    def __init__(self) -> None:
        super(StandardTransformer, self).__init__()
        
        self.mean = None
        self.std = None
        self.valid = None
        self.normalizer = MinMaxTransformer()
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> StandardTransformer:
        
        self.normalizer.fit(input)
        
        input = np.array(input)
        
        axis = [dim for dim in range(input.ndim - 1)]
        axis = tuple(axis)
        
        self.mean = np.mean(input, axis=axis, keepdims=True)
        self.std = np.std(input, axis=axis, keepdims=True)
        self.valid = np.copy(self.std > 0)
        
        self.state = {"mean": self.mean,
                      "std": self.std,
                      "valid": self.valid}
        
        return self

    def _transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.mean is None or self.std is None or self.valid is None:
            raise ValueError("Transformer not fitted")
        
        if isinstance(input, torch.Tensor):
            mean = torch.from_numpy(self.mean).to(input.device)
            std = torch.from_numpy(self.std).to(input.device)
            valid = torch.from_numpy(self.valid).to(input.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output = torch.where(valid, (input - mean) / std, 0)
        elif isinstance(input, np.ndarray):
            mean = np.array(self.mean)
            std = np.array(self.std)
            valid = np.array(self.valid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output = np.where(valid, (input - mean) / std, 0)
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.mean is None or self.std is None or self.valid is None:
            raise ValueError("Transformer not fitted")
        
        mean = self.mean
        std = self.std
        if isinstance(input, torch.Tensor):
            mean = torch.Tensor(self.mean).to(input.device)
            std = torch.Tensor(self.std).to(input.device)
            output = input * std + mean
        elif isinstance(input, np.ndarray):
            mean = np.array(self.mean)
            std = np.array(self.std)
            output = input * std + mean
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        output = self.detransform(input)
        output = self.normalizer.transform(output)
        return output
    