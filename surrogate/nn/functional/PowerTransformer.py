from __future__ import annotations
from typing import Union

import torch
import numpy as np
import sklearn.preprocessing as pp

from .Transformer import Transformer
from .MinMaxTransformer import MinMaxTransformer

class PowerTransformer(Transformer):
    def __init__(self,
                **kwargs) -> None:
        super(PowerTransformer, self).__init__()
        
        self.transformer = pp.PowerTransformer(**kwargs)
        self.normalizer = MinMaxTransformer()
    
    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> PowerTransformer:
        
        self.normalizer.fit(input)
        
        input = input.reshape((-1, input.shape[-1]))
        self.transformer.fit(input)
        self.state = {"transformer": self.transformer}
        return self

    def _transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if len(self.state) == 0:
            raise ValueError("Transformer not fitted")
        
        orig_shape = input.shape
        input = input.reshape((-1, input.shape[-1]))
        output = self.transformer.transform(input)
        output = output.reshape(orig_shape)
        
        if isinstance(input, torch.Tensor):
            output = torch.from_numpy(output).to(input.device)
        
        return output

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if len(self.state) == 0:
            raise ValueError("Transformer not fitted")
        
        orig_shape = input.shape
        input = input.reshape((-1, input.shape[-1]))
        output = self.transformer.inverse_transform(input)
        output = output.reshape(orig_shape)
        
        if isinstance(input, torch.Tensor):
            output = torch.from_numpy(output).to(input.device)
        
        return output
    
    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        output = self.detransform(input)
        output = self.normalizer.transform(output)
        return output
    