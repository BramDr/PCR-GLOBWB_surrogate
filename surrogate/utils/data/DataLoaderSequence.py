from typing import Optional
import torch.utils.data as data

import torch
import torch.utils.data as data

from .RecurrentBatchsetSequence import RecurrentBatchsetSequence


class DataLoaderSequence():
    def __init__(self,
                 dataset: RecurrentBatchsetSequence,
                 sampler: Optional[data.Sampler] = None) -> None:

        if sampler is None:
            sampler = data.SequentialSampler(data_source=dataset)

        self.sampler = sampler
        self.dataset = dataset
        self.iter = iter([])

    def __str__(self):
        string = "{} with {}".format(type(self).__name__,
                                     self.dataset)
        return string

    def __iter__(self):
        self.iter = iter(self.sampler)
        return self

    def __next__(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        try:
            index = next(self.iter)
        except StopIteration:
            raise StopIteration

        return self.dataset[index]
