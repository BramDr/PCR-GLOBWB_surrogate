from __future__ import annotations
from typing import Optional, Sequence
import copy

import numpy as np

import sklearn.preprocessing as pp

from .Transformer import Transformer
from .Flattener import Flattener

class SklearnTransformer(Transformer):
    def __init__(self,
                 transformer: pp.QuantileTransformer,
                 flattener: Optional[Flattener] = None) -> None:
        
        super(SklearnTransformer, self).__init__()
        
        if flattener is None:
            flattener =  Flattener(reset_state=True)
        
        self.sklearner = transformer
        self.flattener = flattener
    
    def fit(self,
            input: np.ndarray) -> SklearnTransformer:
        
        self.flattener.fit(input)
        input = self.flattener.transform(input)
        self.sklearner.fit(input)
        
        self.state = {"flattener": self.flattener,
                      "sklearner": self.sklearner}
        
        return self

    def _transform(self, input: np.ndarray) -> np.ndarray:
        input = self.flattener.transform(input)
        output = self.sklearner.transform(input)
        output = self.flattener.detransform(output)
        return output

    def detransform(self, input: np.ndarray) -> np.ndarray:
        input = self.flattener.transform(input)     
        output = self.sklearner.inverse_transform(input)
        output = self.flattener.detransform(output)        
        return output
    