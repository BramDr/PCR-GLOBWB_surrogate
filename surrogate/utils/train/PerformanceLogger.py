import pathlib as pl
import csv

from .Callback import Callback


class PerformanceLogger(Callback):
    def __init__(self,
                 path: pl.Path,
                 delayed: bool = True,
                 verbose: bool = False) -> None:
        super(PerformanceLogger, self).__init__()

        self.path = pl.Path("{}/performance.csv".format(path))
        self.delayed = delayed
        self.verbose = verbose
        
        self.initialized = False
        self.performances = None

    def epoch_end_callback(self, trainer) -> None:
        statistics_dict = dict(zip(["epoch",
                                    "train loss",
                                    "validation loss",
                                    "best loss"], [trainer.epoch,
                                                   trainer.train_loss,
                                                   trainer.validation_loss,
                                                   trainer.best_loss]))
        
        if self.performances is None:
            self.performances = []
        
        if self.delayed:
            self.performances.append(statistics_dict)
            return

        self.performances = [statistics_dict]
        self.write()

    def end_callback(self, trainer) -> None:
        if not self.delayed:
            return

        self.write()
    
    def write(self):
        if self.performances is None:
            raise ValueError("Saving performance before training has begun.")
        
        if self.verbose:
            print("Writing performance to {}".format(self.path))
        
        fieldnames = self.performances[0].keys()
        
        if not self.initialized:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w', newline='') as f:
                dict_object = csv.DictWriter(f, fieldnames=fieldnames)
                dict_object.writeheader()
                self.initialized = True
            
        with open(self.path, 'a', newline='') as file:
            dict_object = csv.DictWriter(file, fieldnames=fieldnames)
            for performance in self.performances:
                dict_object.writerow(performance)
        
