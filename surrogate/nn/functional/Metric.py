from typing import Callable,  Optional, Sequence

import torch

metric_fn_t = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
reduce_fn_t = Callable[[torch.Tensor], torch.Tensor]


class Metric():
    def __init__(self,
                 metric_fn: metric_fn_t,
                 pre_reduce_fns: Optional[Sequence[reduce_fn_t]] = None,
                 post_reduce_fns: Optional[Sequence[reduce_fn_t]] = None) -> None:
        
        if pre_reduce_fns is None:
            pre_reduce_fns = []
        if post_reduce_fns is None:
            post_reduce_fns = []
        
        self.pre_reduce_fns = pre_reduce_fns
        self.metric_fn = metric_fn
        self.post_reduce_fns = post_reduce_fns

    def calculate(self,
                  pred: torch.Tensor,
                  true: torch.Tensor) -> torch.Tensor:
        value = self.metric_fn(pred, true)
        return value

    def pre_reduce(self,
               tensor: torch.Tensor) -> torch.Tensor:
        for pre_reduce_fn in self.pre_reduce_fns:
            tensor = pre_reduce_fn(tensor)
        return tensor

    def post_reduce(self,
               tensor: torch.Tensor) -> torch.Tensor:
        for post_reduce_fn in self.post_reduce_fns:
            tensor = post_reduce_fn(tensor)
        return tensor

    def calculate_reduce(self,
                         pred: torch.Tensor,
                         true: torch.Tensor) -> torch.Tensor:
        pred = self.pre_reduce(tensor=pred)
        value = self.calculate(pred=pred,
                               true=true)
        value = self.post_reduce(tensor=value)
        return value
