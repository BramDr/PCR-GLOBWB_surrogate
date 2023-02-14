import pathlib as pl
import pickle

from .TorchModuleLogger import TorchModuleLogger
from .BestLoss import BestLoss
from .Logger import Logger


class BestModelLogger(TorchModuleLogger, BestLoss):
    def __init__(self,
                 path: pl.Path,
                 delayed: bool = True,
                 verbose: bool = False) -> None:
        super(BestModelLogger, self).__init__(path=path)

        self.delayed = delayed
        self.verbose = verbose
        self.best_state_dict = None
        self.meta = None

    def epoch_end_callback(self, trainer, epoch: int, statistics: dict) -> None:
        if not super().assess_loss(statistics=statistics):
            return

        if self.verbose:
            print("New checkpoint {: .3e}".format(self.best_loss), flush=True)

        self.best_state_dict = trainer.model.state_dict()
        self.meta = {"in_features": trainer.model.in_features,
                     "out_features": trainer.model.out_features}

        if self.delayed:
            return

        if self.verbose:
            print("Writing model to {}".format(self.path), flush=True)

        self.write(state_dict=self.best_state_dict)
        #print(self.best_state_dict["pre_sequence.0.weight"])   

    def end_callback(self, trainer) -> None:
        if not self.delayed:
            return

        if self.verbose:
            print("Writing model to {}".format(self.path), flush=True)

        if self.best_state_dict is not None:
            self.write(state_dict=self.best_state_dict)
            #print(self.best_state_dict["pre_sequence.0.weight"])   
