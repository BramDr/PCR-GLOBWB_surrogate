from typing import Optional
import numpy as np

from surrogate.nn.SurrogateModelSelfUpstream import SurrogateModel
from surrogate.utils.correct.Correcter import Correcter


def build_correcter(true: np.ndarray,
                    pred: np.ndarray,
                    max_values: Optional[int] = None,
                    verbose: int = 1) -> Correcter:

    corrector = Correcter(true=true,
                          pred=pred,
                          max_values=max_values)

    if verbose > 0:
        print("Loaded {}: ({} {})".format(type(corrector).__name__,
                                          corrector.true.shape,
                                          corrector.pred.shape))

    return corrector
