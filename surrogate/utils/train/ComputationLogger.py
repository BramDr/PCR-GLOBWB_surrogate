import time
import pathlib as pl

from .CsvDictLogger import CsvDictLogger
from .Callback import Callback


class ComputationLogger(CsvDictLogger, Callback):
    def __init__(self,
                 path: pl.Path,
                 verbose: bool = False) -> None:
        super(ComputationLogger, self).__init__(path=path)

        self.loader = None
        self.verbose = verbose

    def start_callback(self,
                       trainer) -> None:
        try:
            self.loader = trainer.test_loaders[0]
        except IndexError:
            self.loader = trainer.train_loader

    def end_callback(self,
                     trainer) -> None:
        model = trainer.model

        wall_time = time.time()
        cpu_time = time.perf_counter()

        n_samples = 0
        n_sequence = 0
        for x, _ in self.loader:
            n_samples += x.shape[0]
            n_sequence = x.shape[1]
            _, _ = model.forward(x)

        cpu_time = time.perf_counter() - cpu_time
        wall_time = time.time() - wall_time

        cpu_time /= n_samples * n_sequence  # sec per 1000 samples per 1000 timesteps
        wall_time /= n_samples * n_sequence  # sec per 1000 samples per 1000 timesteps

        if self.verbose:
            print("Writing time to {}".format(self.path), flush=True)

        self.write_dict([{"cpu": cpu_time, "wall": wall_time}])
