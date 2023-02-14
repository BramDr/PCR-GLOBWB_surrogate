from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

from .Transformer import Transformer


class Flattener(Transformer):
    def __init__(self,
                 reset_state: bool = False) -> None:
        
        super().__init__()
        
        self.reset_state = reset_state

    def fit(self,
            input: np.ndarray) -> Flattener:
        
        shape = input.shape
        flat_shape = (-1, input.shape[-1])

        self.state["shape"] = shape
        self.state["flat_shape"] = flat_shape
        
        return self

    def _transform(self,
                   input: np.ndarray) -> np.ndarray:
        
        if self.reset_state:
            self.fit(input)
            
        flat_shape = self.state["flat_shape"]
        input = np.reshape(input, flat_shape)

        return input

    def detransform(self, input: np.ndarray) -> np.ndarray:
        
        shape = self.state["shape"]
        input = np.reshape(input, shape)

        return input
