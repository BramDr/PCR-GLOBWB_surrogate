import pathlib as pl
import copy

from .Callback import Callback


class SurrogateModelLogger(Callback):
    def __init__(self,
                 path: pl.Path,
                 delayed: bool = True,
                 verbose: bool = False) -> None:
        super(SurrogateModelLogger, self).__init__()

        self.path = path
        self.delayed = delayed
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.best_model = None

    def epoch_end_callback(self, trainer) -> None:
        if trainer.best_loss >= self.best_loss:
            return
        self.best_loss = trainer.best_loss
        
        if self.verbose:
            print("New checkpoint {: .3e}".format(self.best_loss))
        self.best_model = copy.deepcopy(trainer.model)
        
        if self.delayed:
            return
        self.write()

    def end_callback(self, trainer) -> None:
        if not self.delayed:
            return

        self.write()
            
    def write(self) -> None:
        if self.best_model is None:
            raise ValueError("Saving model before training has begun.")
        
        if self.verbose:
            print("Writing model to {}".format(self.path))
        self.best_model.save(self.path)
