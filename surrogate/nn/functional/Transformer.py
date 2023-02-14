from __future__ import annotations
from typing import Optional, Sequence
from abc import ABC
import copy

import numpy as np


class Transformer(ABC):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
            
        self.state = {}

    def fit(self,
            input: np.ndarray) -> Transformer:
        raise NotImplementedError("transform not implemented")

    def _transform(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("transform not implemented")

    def transform(self, input: np.ndarray) -> np.ndarray:
        if self.state == {}:
            self.fit(input=input)
        return self._transform(input=input)

    def detransform(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("detransform not implemented")
    
    def __str__(self):
        string = "{}".format(type(self).__name__)
        return string