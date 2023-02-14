from typing import Union, Optional, Sequence

import pathlib as pl
import torch.optim as optim

from surrogate.nn import SurrogateModelSelfUpstream
from surrogate.nn.functional import Metric
from surrogate.nn.functional import SequenceMseMetric
from surrogate.utils.data import SequenceDataLoader
from surrogate.utils.data import SequencePreFetcher

from .BestModelLogger import BestModelLogger
from .PerformanceLogger import PerformanceLogger


def build_trainer(dataloader: Union[SequenceDataLoader, SequencePreFetcher],
                  model: SurrogateModelSelfUpstream,
                  dir_out: pl.Path = pl.Path("./"),
                  statistics_callback: bool = False,
                  state_dict_callback: bool = False,
                  delayed: bool = True,
                  callbacks: Optional[list] = None,
                  loss: Metric = SequenceMseMetric(),
                  test_dataloaders: Optional[Sequence[Union[SequenceDataLoader, SequencePreFetcher]]] = None,
                  learning_rate: float = 1e-2,
                  weight_decay: float = 0,
                  verbose: int = 1) -> CallbackModuleTrainer:
    
    if callbacks is None:
        callbacks = []
    if test_dataloaders is None:
        test_dataloaders = []

    if statistics_callback:
        callback_out = pl.Path("{}/statistics.csv".format(dir_out))
        callback = PerformanceLogger(path=callback_out,
                                     delayed=delayed)
        callbacks.append(callback)
    if state_dict_callback:
        callback_out = pl.Path("{}/state_dict.pt".format(dir_out))
        callback = BestModelLogger(path=callback_out,
                                   delayed=delayed)
        callbacks.append(callback)

    if verbose > 0:
        for callback in callbacks:
            print("Setup callback {}".format(
                type(callback).__name__), flush=True)

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if verbose > 0:
        print(optimizer, flush=True)

    trainer = CallbackModuleTrainer(train_loader=dataloader,
                                    model=model,
                                    loss=loss,
                                    optimizer=optimizer,
                                    callbacks=callbacks,
                                    test_dataloaders=test_dataloaders)

    return trainer
