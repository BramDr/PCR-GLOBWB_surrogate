from typing import Union

import torch
import numpy as np


class Metric():
    def __init__(self,
                 reduction: str = "mean"):
        self.reduction = reduction
        
    def forward(self,
                input: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError()
        
    def __call__(self,
                 input: Union[torch.Tensor, np.ndarray],
                 target: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return self.forward(input, target)
    
    def _reduce(self,
                value: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        
        if isinstance(value, torch.Tensor):
            value[torch.isinf(value)] = torch.nan
            if self.reduction == "none":
                pass
            elif self.reduction == "mean":
                value = torch.nanmean(value)
            elif self.reduction == "median":
                value = torch.nanmedian(value)
            elif self.reduction == "sum":
                value = torch.nansum(value)
            else:
                raise ValueError("Unknown reduction {}".format(self.reduction))
        elif isinstance(value, np.ndarray):
            value[np.isinf(value)] = np.nan
            if self.reduction == "none":
                pass
            elif self.reduction == "mean":
                value = np.nanmean(value)
            elif self.reduction == "median":
                value = np.nanmedian(value)
            elif self.reduction == "sum":
                value = np.nansum(value)
            else:
                raise ValueError("Unknown reduction {}".format(self.reduction))
            
        return value