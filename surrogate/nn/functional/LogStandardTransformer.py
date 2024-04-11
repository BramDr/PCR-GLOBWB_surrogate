from __future__ import annotations
from functools import partial
from typing import Union
import math

import torch
import numpy as np

from .Transformer import Transformer
from .StandardTransformer import StandardTransformer
from .MinMaxTransformer import MinMaxTransformer

class LogStandardTransformer(Transformer):
    def __init__(self,
                 small: float = 1e-10,
                 base: float = math.e) -> None:
        super(LogStandardTransformer, self).__init__()
                    
        self.small = small
        self.total_min = None
        self.subtransformer = None
        self.normalizer = MinMaxTransformer()
        
        self.base = base
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> LogStandardTransformer:
        
        self.normalizer.fit(input)
        
        input = np.array(input)
        
        self.total_min = np.min(input)
        input = input - self.total_min
        input = np.log(input + self.small) / np.log(self.base)
        
        self.subtransformer = StandardTransformer()
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
            input = torch.log(input + self.small) / torch.log(self.base)
                
        elif isinstance(input, np.ndarray):
            input = np.log(input + self.small) / np.log(self.base)
                
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")
        
        output = self.subtransformer._transform(input = input)
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.total_min is None or self.subtransformer is None:
            raise ValueError("Transformer not fitted")
        
        output = self.subtransformer.detransform(input = input)
        
        if isinstance(input, torch.Tensor):
            output = torch.pow(self.base, output) - self.small
            output = output + self.total_min
            
        elif isinstance(input, np.ndarray):
            output = np.power(self.base, output) - self.small
            output = output + self.total_min
            
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")

        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        output = self.detransform(input)
        output = self.normalizer.transform(output)
        return output
    