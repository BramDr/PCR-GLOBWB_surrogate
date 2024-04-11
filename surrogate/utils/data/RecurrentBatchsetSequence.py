from typing import Optional

import numpy as np
import pandas as pd
import torch

from surrogate.nn.functional import Transformer

from .RecurrentBatchset import RecurrentBatchset


class RecurrentBatchsetSequence(RecurrentBatchset):
    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 samples_size: Optional[int] = None,
                 dates_size: Optional[int] = None,
                 samples: Optional[np.ndarray] = None,
                 lons: Optional[np.ndarray] = None,
                 lats: Optional[np.ndarray] = None,
                 dates: Optional[np.ndarray] = None,
                 x_features: Optional[np.ndarray] = None,
                 y_features: Optional[np.ndarray] = None,
                 x_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                 y_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                 squeeze_batch: bool = False,
                 copy_batch: bool = True,
                 drop_last: bool = False,
                 cuda: bool = False) -> None:
        super(RecurrentBatchsetSequence, self).__init__(x=x,
                                                        y=y,
                                                        samples_size=samples_size,
                                                        dates_size=dates_size,
                                                        samples=samples,
                                                        lons=lons,
                                                        lats=lats,
                                                        dates=dates,
                                                        x_features=x_features,
                                                        y_features=y_features,
                                                        x_transformers=x_transformers,
                                                        y_transformers=y_transformers,
                                                        squeeze_batch=squeeze_batch,
                                                        copy_batch=copy_batch,
                                                        drop_last=drop_last,
                                                        cuda=cuda)

    def __str__(self):
        x, y = self[0]
        
        y_shape = None
        x_shape = None
        if x[0] is not None:
            x_shape = x[0].shape
        if y[0] is not None:
            y_shape = y[0].shape

        string = "{}: {} ({} x {}) ({}, {})".format(type(self).__name__,
                                                        self.batch_len,
                                                        self.mapped_samples_len,
                                                        self.mapped_dates_len,
                                                        x_shape,
                                                        y_shape)
        return string

    def __len__(self):
        return self.mapped_samples_len

    def __getitem__(self,
                    index) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_items = self.x_mapped[index]
        y_items = self.y_mapped[index]

        return x_items, y_items

    def _reconstruct(self,
                     tensors: list[list[torch.Tensor]]) -> torch.Tensor:

        sample_tensors = []
        for sample_index in pd.unique(self.sample_mapping):
            date_tensors = tensors[sample_index]

            sample_tensor = torch.concat(date_tensors, dim=0)
            sample_tensors.append(sample_tensor)

        tensor = torch.concat(sample_tensors, dim=-2)

        return tensor
