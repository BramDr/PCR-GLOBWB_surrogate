from __future__ import annotations
from typing import Union
import abc

import torch
import numpy as np


class Transformer(abc.ABC):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
            
        self.state = {}

    @abc.abstractmethod
    def fit(self,
            array: Union[np.ndarray, torch.Tensor]) -> Transformer:
        pass

    @abc.abstractmethod
    def _transform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _detransform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        pass

    def transform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.state == {}:
            raise ValueError("fit must be called before transform")
        return self._transform(array=array)

    def detransform(self, array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.state == {}:
            raise ValueError("fit must be called before transform")
        return self._detransform(array=array)
    
    def __str__(self):
        return f"{type(self).__name__}"