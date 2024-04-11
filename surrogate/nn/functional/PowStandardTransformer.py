from __future__ import annotations
from functools import partial
from typing import Union

import torch
import numpy as np

from .Transformer import Transformer
from .StandardTransformer import StandardTransformer
from .MinMaxTransformer import MinMaxTransformer

class PowStandardTransformer(Transformer):
    def __init__(self,
                 power: float = 1.0) -> None:
        super(PowStandardTransformer, self).__init__()
                    
        self.total_min = None
        self.subtransformer = None
        self.normalizer = MinMaxTransformer()
        
        self.power = power
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> PowStandardTransformer:
        
        self.normalizer.fit(input)
        
        input = np.array(input)
        
        self.total_min = np.min(input)
        input = input - self.total_min
        input = input**self.power
        
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
        
        input = input**self.power
        
        output = self.subtransformer._transform(input = input)
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.total_min is None or self.subtransformer is None:
            raise ValueError("Transformer not fitted")
        
        output = self.subtransformer.detransform(input = input)
        
        output = output**(1 / self.power)
        output = output + self.total_min

        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        output = self.detransform(input)
        output = self.normalizer.transform(output)
        return output
    