from typing import Optional
import torch.utils.data as data

import torch
import torch.utils.data as data

from .SequenceBatchset import SequenceBatchset

class SequenceDataLoader():
    def __init__(self,
                 dataset: SequenceBatchset,
                 sampler: Optional[data.Sampler] = None) -> None:

        if sampler is None:
            sampler = data.SequentialSampler(data_source=dataset)

        self.sampler = sampler
        self.dataset = dataset
        self.iter = iter([])
        
    def __str__(self):
        x, y = next(iter(self))
        y_shape = None
        x_shape = None
        if x is not None:
            x_shape = x.shape
        if y is not None:
            y_shape = y.shape
            
        string = "{}: ({}, {})".format(type(self).__name__,
                                      x_shape,
                                      y_shape)
        return string

    def __iter__(self):
        self.iter = iter(self.sampler)
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            index = next(self.iter)
        except StopIteration:
            raise StopIteration

        return self.dataset[index]
