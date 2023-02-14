import pathlib as pl

from .CsvDictLogger import CsvDictLogger
from .Callback import Callback


class PerformanceLogger(CsvDictLogger, Callback):
    def __init__(self,
                 path: pl.Path,
                 delayed: bool = True,
                 verbose: bool = False) -> None:
        super(PerformanceLogger, self).__init__(path=path)

        self.delayed = delayed
        self.verbose = verbose
        self.performances = []

    def epoch_end_callback(self, trainer, epoch: int, statistics: dict) -> None:
        if self.delayed:
            self.performances.append(statistics)
            return

        if self.verbose:
            print("Writing statistics to {}".format(self.path), flush=True)

        self.write_dict([statistics])

    def end_callback(self, trainer) -> None:
        if not self.delayed:
            return

        if self.verbose:
            print("Writing statistics to {}".format(self.path), flush=True)

        self.write_dict(self.performances)
