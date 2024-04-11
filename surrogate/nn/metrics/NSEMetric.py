import torch

from .Metric import Metric


class NSEMetric(Metric):
    def __init__(self,
                 reduction: str = "mean"):
        super(NSEMetric, self).__init__(reduction=reduction)
        
    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        
        top = torch.sum((target - input)**2, dim = 0, keepdim = True)
        bottom = torch.sum((target - torch.mean(target, dim = 0, keepdim = True))**2, dim = 0, keepdim = True)
        
        nse = 1 - top / bottom
            
        return self._reduce(nse)
        