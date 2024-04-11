from typing import Union

import torch
import numpy as np

from .Metric import Metric


class KGEMetric(Metric):
    def __init__(self,
                 reduction: str = "mean"):
        super(KGEMetric, self).__init__(reduction=reduction)
        
    def forward(self,
                input: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        
        if isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor):
            r_input_err = input - torch.mean(input, dim = 0, keepdim = True)
            r_target_err = target - torch.mean(target, dim = 0, keepdim = True)
            r = torch.sum(r_input_err * r_target_err, dim = 0, keepdim = True) / (torch.sqrt(torch.sum(r_input_err**2)) * torch.sqrt(torch.sum(r_target_err**2)))
            alpha = torch.var(input, dim = 0, keepdim = True) / torch.var(target, dim = 0, keepdim = True)
            beta = torch.mean(input, dim = 0, keepdim = True) / torch.mean(target, dim = 0, keepdim = True)
            
            kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
        elif isinstance(input, np.ndarray)  and isinstance(target, np.ndarray):
            r_input_err = input - np.mean(input, axis = 0, keepdims = True)
            r_target_err = target - np.mean(target, axis = 0, keepdims = True)
            r = np.sum(r_input_err * r_target_err, axis = 0, keepdims = True) / (np.sqrt(np.sum(r_input_err**2)) * np.sqrt(np.sum(r_target_err**2)))
            alpha = np.var(input, axis = 0, keepdims = True) / np.var(target, axis = 0, keepdims = True)
            beta = np.mean(input, axis = 0, keepdims = True) / np.mean(target, axis = 0, keepdims = True)
            
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        else:
            raise ValueError("Input type {} and target type {} do not match".format(type(input),
                                                                                    type(target)))
        
        return self._reduce(kge)
        