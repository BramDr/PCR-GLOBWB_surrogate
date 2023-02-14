import numpy as np

from .Transformer import Transformer


class Sequencer(Transformer):
    def __init__(self,
                 transformers: list[Transformer]) -> None:
        super().__init__()
        self.transformers = transformers

    def fit(self,
            input: np.ndarray) -> None:
        for transformer in self.transformers:
            transformer.fit(input)
        self.state = {"transformers": self.transformers}

    def _transform(self,
                   input: np.ndarray) -> np.ndarray:
        for transformer in self.transformers:
            input = transformer._transform(input)
        return input

    def detransform(self, input: np.ndarray) -> np.ndarray:
        for transformer in self.transformers:
            input = transformer.detransform(input)
        return input
