from typing import Optional
import numpy as np
import scipy.stats as stats

from .Transformer import Transformer
from .Flattener import Flattener


class Logger(Transformer):
    def __init__(self,
                 skew_threshold: Optional[int] = 5,
                 skew_error: Optional[float] = 1e-12,
                 flattener: Optional[Flattener] = None) -> None:
        super().__init__()

        if flattener is None:
            flattener = Flattener()

        self.skew_threshold = skew_threshold
        self.skew_error = skew_error
        self.flattener = flattener

    def fit(self,
            input: np.ndarray) -> None:
        input = self.flattener.transform(input)
        skewness = stats.skew(input, axis=0)
        skewness = np.expand_dims(skewness, axis=0)
        self.state = {"skewness": skewness}

    def _transform(self,
                   input: np.ndarray) -> np.ndarray:
        input = self.flattener.transform(input)
        skewness = self.state["skewness"]
        skewness = np.broadcast_to(skewness, input.shape)
        output = np.log(input + self.skew_error, where=skewness >
                        self.skew_threshold, out=np.copy(input))
        output = self.flattener.detransform(output)
        return output

    def detransform(self, input: np.ndarray) -> np.ndarray:
        input = self.flattener.transform(input)
        skewness = self.state["skewness"]
        skewness = np.broadcast_to(skewness, input.shape)
        output = np.exp(input - self.skew_error, where=skewness >
                        self.skew_threshold, out=np.copy(input))
        output = self.flattener.detransform(output)
        return output
