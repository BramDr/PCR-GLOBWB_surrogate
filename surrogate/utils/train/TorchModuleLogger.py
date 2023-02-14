import pathlib as pl
import torch

from .Logger import Logger


class TorchModuleLogger(Logger):
    def __init__(self,
                 path: pl.Path) -> None:
        super(TorchModuleLogger, self).__init__(path=path)

    def write(self, state_dict: dict) -> None:
        super(TorchModuleLogger, self).write()

        torch.save(state_dict, self.path)
