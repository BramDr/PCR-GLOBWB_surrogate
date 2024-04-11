from __future__ import annotations
from typing import Union
from abc import ABC

import torch
import numpy as np


class Transformer(ABC):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
            
        self.state = {}

    def fit(self,
            input: Union[np.ndarray, torch.Tensor]) -> Transformer:
        raise NotImplementedError("transform not implemented")

    def _transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError("transform not implemented")

    def transform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.state == {}:
            self.fit(input=input)
        return self._transform(input=input)

    def detransform(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError("detransform not implemented")

    def detransform_normalize(self, input: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError("detransform not implemented")
    
    def __str__(self):
        string = "{}".format(type(self).__name__)
        return string