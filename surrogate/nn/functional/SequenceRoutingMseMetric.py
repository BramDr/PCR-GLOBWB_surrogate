from typing import Callable, Optional

import torch

from .metrics import SE
from .Metric import Metric

reduce_fn_t = Callable[[torch.Tensor], torch.Tensor]


class SequenceRoutingMseMetric(Metric):
    def __init__(self,
                 sequence_reduce_fn: Optional[reduce_fn_t] = None,
                 sample_reduce_fn: Optional[reduce_fn_t] = None,
                 feature_reduce_fn: Optional[reduce_fn_t] = None) -> None:
        metric_fn = SE

        def sample_reduce_fn_def(x): return torch.sum(x, dim=0)
        def sequence_reduce_fn_def(x): return torch.mean(x, dim=0, keepdim=True)
        def feature_reduce_fn_def(x): return torch.mean(x, dim=1, keepdim=True)
        
        if sample_reduce_fn is None:
            sample_reduce_fn = sample_reduce_fn_def
        if sequence_reduce_fn is None:
            sequence_reduce_fn = sequence_reduce_fn_def
        if feature_reduce_fn is None:
            feature_reduce_fn = feature_reduce_fn_def

        pre_reduce_fns = [sample_reduce_fn]
        post_reduce_fns = [sequence_reduce_fn, feature_reduce_fn]

        super(SequenceRoutingMseMetric, self).__init__(metric_fn=metric_fn,
                                                       pre_reduce_fns=pre_reduce_fns,
                                                       post_reduce_fns=post_reduce_fns)
