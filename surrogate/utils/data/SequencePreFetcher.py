from typing import Optional
import time

import torch
import torch.cuda as cuda

from surrogate.utils.data import SequenceDataLoader


class SequencePreFetcher():
    def __init__(self,
                 dataloader: SequenceDataLoader):
        self.loader = dataloader
        self.dataset = dataloader.dataset
        self.stream = cuda.Stream()
        self.iter = None
        self.next_x = None
        self.next_y = None

        self._get_next = 0.
        self._gpu_stream = 0.
        
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

    def _preload(self) -> None:
        try:
            #get_start = time.perf_counter()
            self.next_x, self.next_y = next(self.iter)
            #get_time = time.perf_counter() - get_start
            #print("Get time {: .2e}".format(get_time))
        except StopIteration:
            self.next_x = None
            self.next_y = None
            return

        with cuda.stream(self.stream):
            #stream_start = time.perf_counter()
            self.next_x = self.next_x.cuda(non_blocking=True)
            self.next_y = self.next_y.cuda(non_blocking=True)
            #stream_time = time.perf_counter() - stream_start
            #print("Stream time {: .2e}".format(stream_time))

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cuda.current_stream().wait_stream(self.stream)

        x = self.next_x
        y = self.next_y

        if x is None and y is None:
            raise StopIteration()

        x.record_stream(cuda.current_stream())
        y.record_stream(cuda.current_stream())
            
        self._preload()
        return x, y
