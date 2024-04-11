import torch
import torch.cuda as cuda

from .DataLoaderSequence import DataLoaderSequence


class PreFetcherSequence():
    def __init__(self,
                 dataloader: DataLoaderSequence,
                 gpu = 0):
        self.loader = dataloader
        self.dataset = dataloader.dataset
        self.stream = cuda.Stream(gpu)
        self.iter = iter([])
        self.next_xs = None
        self.next_ys = None
        self.gpu = gpu

        self._get_next = 0.
        self._gpu_stream = 0.

    def __str__(self):
        string = "{} with {}".format(type(self).__name__,
                                     self.dataset)
        return string

    def _preload(self) -> None:
        try:
            self.next_xs, self.next_ys = next(self.iter)
        except StopIteration:
            self.next_xs = None
            self.next_ys = None
            return

        with cuda.stream(self.stream):
            self.next_xs = [x.cuda(self.gpu, non_blocking=True) if x is not None else None for x in self.next_xs]
            self.next_ys = [y.cuda(self.gpu, non_blocking=True) if y is not None else None for y in self.next_ys]

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cuda.current_stream(self.gpu).wait_stream(self.stream)

        xs = self.next_xs
        ys = self.next_ys

        if xs is None or ys is None:
            raise StopIteration()

        [x.record_stream(cuda.current_stream(self.gpu)) for x in xs if x is not None]
        [y.record_stream(cuda.current_stream(self.gpu)) for y in ys if y is not None]

        self._preload()
        return xs, ys
