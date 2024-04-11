from typing import Optional
import pathlib as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import numpy as np

from surrogate.utils.data import load_batchset_sequence
from surrogate.utils.data import load_dataloader_sequence
from surrogate.nn import SurrogateModel
from surrogate.utils.train import PerformanceLogger
from surrogate.utils.train import SurrogateModelLogger
from surrogate.utils.train import ModuleTrainerSequence

def train_objective(n_lstm: int,
                    n_in_linear: int,
                    n_out_linear: int,
                    in_hidden_size: int,
                    out_hidden_size: int,
                    dropout_rate: float,
                    samples_size: int,
                    dates_size: int,
                    learning_rate: float,
                    epochs: int,
                    train_dir: pl.Path,
                    validation_dir: pl.Path,
                    test_dir: pl.Path,
                    transformer_dir: pl.Path,
                    out_dir: pl.Path,
                    input_features: np.ndarray,
                    output_features: np.ndarray,
                    train_split_fraction: float = 1.0,
                    validation_split_fraction: float = 1.0,
                    test_split_fraction: float = 1.0,
                    learning_rate_loops: int = 1,
                    final_learning_rate_fraction: float = 1.0,
                    learning_rate_maximum: Optional[float] = None,
                    cuda: bool = False,
                    seed: int = 19920223,
                    verbose: int = 1) -> ModuleTrainerSequence:
    
    if learning_rate_maximum is None:
        learning_rate_maximum = learning_rate

    # Data
    train_batchset = load_batchset_sequence(array_dir=train_dir,
                                            meta_dir=train_dir,
                                            transformer_dir=transformer_dir,
                                            input_features=input_features,
                                            output_features=output_features,
                                            samples_size=samples_size,
                                            dates_size=dates_size,
                                            cuda = cuda,
                                            split_fraction=train_split_fraction,
                                            seed=seed,
                                            verbose=verbose - 1)
    train_generator = torch.Generator().manual_seed(seed)
    train_sampler = data.RandomSampler(data_source=train_batchset,
                                       generator=train_generator)
    train_dataloader = load_dataloader_sequence(dataset=train_batchset,
                                                sampler=train_sampler,
                                                allow_prefetcher=not cuda,
                                                verbose=verbose - 1)
    
    validation_batchset = load_batchset_sequence(array_dir=validation_dir,
                                                 meta_dir=validation_dir,
                                                 transformer_dir=transformer_dir,
                                                 input_features=input_features,
                                                 output_features=output_features,
                                                 samples_size=samples_size,
                                                 dates_size=dates_size,
                                                 cuda=cuda,
                                                 split_fraction=validation_split_fraction,
                                                 seed=seed,
                                                 verbose=verbose - 1)
    validation_generator = torch.Generator().manual_seed(seed)
    validation_sampler = data.RandomSampler(data_source=validation_batchset,
                                            generator=validation_generator)
    validation_dataloader = load_dataloader_sequence(dataset=validation_batchset,
                                                     sampler=validation_sampler,
                                                     allow_prefetcher=not cuda,
                                                     verbose=verbose - 1)
    
    test_batchset = load_batchset_sequence(array_dir=test_dir,
                                           meta_dir=test_dir,
                                           transformer_dir=transformer_dir,
                                           input_features=input_features,
                                           output_features=output_features,
                                           samples_size=samples_size,
                                           dates_size=dates_size,
                                           cuda=cuda,
                                           split_fraction=test_split_fraction,
                                           seed=seed,
                                           verbose=verbose - 1)
    test_dataloader = load_dataloader_sequence(dataset=test_batchset,
                                               allow_prefetcher=not cuda,
                                               verbose=verbose - 1)
    
    # Model
    model = SurrogateModel(input_size=input_features.size,
                           output_size=output_features.size,
                           n_lstm=n_lstm,
                           in_hidden_size=[in_hidden_size] * n_in_linear,
                           out_hidden_size=[out_hidden_size] * n_out_linear,
                           dropout_rate=dropout_rate,
                           input_features=input_features,
                           output_features=output_features,
                           input_transformers=train_batchset.x_transformers,
                           output_transformers=train_batchset.y_transformers,
                           try_cuda=True,
                           seed=seed)
    model.train(mode=False)
    if verbose > 0:
        print(model)
        
    # Trainer
    performance_callback = PerformanceLogger(path=out_dir, delayed=False)
    model_callback = SurrogateModelLogger(path=out_dir, delayed=False, verbose=True)
    callbacks = [performance_callback, model_callback]
    
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=learning_rate)
    gamma = final_learning_rate_fraction ** (1 / epochs)
    scheduler = lr_scheduler.CyclicLR(optimizer=optimizer,
                                      base_lr=learning_rate,
                                      max_lr=learning_rate_maximum,
                                      step_size_up=int(epochs / learning_rate_loops * (1/3)),
                                      step_size_down=int(epochs / learning_rate_loops * (2/3)),
                                      cycle_momentum=False,
                                      mode="exp_range",
                                      gamma=gamma)
        
    trainer = ModuleTrainerSequence(train_loader=train_dataloader,
                                    validation_loader=validation_dataloader,
                                    test_loader=test_dataloader,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    callbacks=callbacks)
    loss = nn.MSELoss(reduction = "mean")
    if verbose > 0:
        print(trainer)
    
    # Run
    if verbose > 0:
        print("Runnning {} epochs".format(epochs))
    train_loss = trainer.run(epochs=epochs,
                             loss = loss,
                             seed=seed)
    if verbose > 0:
        print("Finished training with {}".format(train_loss))
        
    metrics = [loss]
    test_losses = trainer.test(metrics = metrics)
    if verbose > 0:
        print("Finished testing with {}".format(test_losses[0]))
        
    return trainer
    