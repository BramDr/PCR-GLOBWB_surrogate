from typing import Union

import torch
import numpy as np

from .Metric import Metric


class KGEAlphaMetric(Metric):
    def __init__(self,
                 reduction: str = "mean"):
        super(KGEAlphaMetric, self).__init__(reduction=reduction)
        
    def forward(self,
                input: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        
        if isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor):
            alpha = torch.var(input, dim = 0, keepdim = True) / torch.var(target, dim = 0, keepdim = True)
            
        elif isinstance(input, np.ndarray)  and isinstance(target, np.ndarray):
            alpha = np.var(input, axis = 0, keepdims = True) / np.var(target, axis = 0, keepdims = True)
        
        else:
            raise ValueError("Input type {} and target type {} do not match".format(type(input),
                                                                                    type(target)))
        
        return self._reduce(alpha)
        