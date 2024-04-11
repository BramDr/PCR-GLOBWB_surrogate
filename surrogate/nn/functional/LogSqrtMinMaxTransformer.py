from __future__ import annotations
from functools import partial
from typing import Union

import torch
import numpy as np

from .Transformer import Transformer
from .MinMaxTransformer import MinMaxTransformer
from .MinMaxTransformer import MinMaxTransformer

class LogSqrtMinMaxTransformer(Transformer):
    def __init__(self,
                 small: float = 1e-10,
                 add_log: bool = False,
                 log_10: bool = False,
                 add_sqrt: bool = False) -> None:
        super(LogSqrtMinMaxTransformer, self).__init__()
        
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
        self.subtransformer = None
        self.normalizer = MinMaxTransformer()
        
        self.add_sqrt = add_sqrt
        self.add_log = add_log
        self.np_log_fn = np_log_fn
        self.torch_log_fn = torch_log_fn
        self.np_pow_fn = np_pow_fn
        self.torch_pow_fn = torch_pow_fn
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> LogSqrtMinMaxTransformer:
        
        self.normalizer.fit(input)
        
        input = np.array(input)
        
        self.total_min = np.min(input)
        input = input - self.total_min
        if self.add_sqrt:
            input = np.sqrt(input)
        if self.add_log:
            input = self.np_log_fn(input + self.small)
        
        self.subtransformer = MinMaxTransformer()
        self.subtransformer = self.subtransformer.fit(input=input)
        
        self.state = {"subtransformer": self.subtransformer,
                      "total_min": self.total_min}
        
        return self

    def _transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.total_min is None or self.subtransformer is None:
            raise ValueError("Transformer not fitted")
        
        input = input - self.total_min
        input[input < 0] = 0
        
        if isinstance(input, torch.Tensor):
            if self.add_sqrt:
                input = torch.sqrt(input)
            if self.add_log:
                input = self.torch_log_fn(input + self.small)
                
        elif isinstance(input, np.ndarray):
            if self.add_sqrt:
                input = np.sqrt(input)
            if self.add_log:
                input = self.np_log_fn(input + self.small)
                
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        output = self.subtransformer._transform(input = input)
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.total_min is None or self.subtransformer is None:
            raise ValueError("Transformer not fitted")
        
        output = self.subtransformer.detransform(input = input)
        
        if isinstance(input, torch.Tensor):
            if self.add_log:
                output = self.torch_pow_fn(output) - self.small
            if self.add_sqrt:
                output = output**2
            output = output + self.total_min
            
        elif isinstance(input, np.ndarray):
            if self.add_log:
                output = self.np_pow_fn(output) - self.small
            if self.add_sqrt:
                output = output**2
            output = output + self.total_min
            
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")

        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        output = self.detransform(input)
        output = self.normalizer.transform(output)
        return output
    