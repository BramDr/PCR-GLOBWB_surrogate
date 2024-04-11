from typing import Optional, Sequence
import pathlib as pl
import optuna as op

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from surrogate.utils.data import load_batchset_sequence
from surrogate.utils.data import load_dataloader_sequence
from surrogate.nn import SurrogateModel
from surrogate.utils.train import ModuleTrainerSequence
from surrogate.nn.metrics import Metric

def optuna_objective(trial: op.trial.Trial,
                    epochs: int,
                    train_dir: pl.Path,
                    validation_dir: pl.Path,
                    test_dir: pl.Path,
                    transformer_dir: pl.Path,
                    split_fraction: float,
                    seed: int,
                    input_features: Optional[np.ndarray] = None,
                    output_features: Optional[np.ndarray] = None,
                    test_metrics: Optional[Sequence[Metric]] = None,
                    cuda: bool = True,
                    verbose: int = 1) -> float:
    
    n_lstm = trial.suggest_int("n_lstm", 1, 2)
    n_in_linear = trial.suggest_int("n_in_linear", 1, 2)
    n_out_linear = trial.suggest_int("n_out_linear", 1, 2)
    in_hidden_size = trial.suggest_int("in_hidden_size", 32, 1024)
    out_hidden_size = trial.suggest_int("out_hidden_size", 32, 1024)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.8)
    samples_size = trial.suggest_int("sample_size", 1, 64)
    dates_size = trial.suggest_int("dates_size", 32, 1024)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    # transformer_type = trial.suggest_categorical("transformer_type", ["min-max", "standard",
    #                                                                   "sqrt_min-max", "sqrt_standard",
    #                                                                   "log_min-max", "log_standard",
    #                                                                   "log-sqrt_min-max", "log-sqrt_standard",
    #                                                                   "log10_min-max", "log10_standard",
    #                                                                   "log10-sqrt_min-max", "log10-sqrt_standard"])
    transformer_type = trial.suggest_categorical("transformer_type", ["standard", "sqrt_standard",
                                                                      "log_standard", "log-sqrt_standard",
                                                                      "losg-0p001_standard", "losg-0p01_standard",
                                                                      "losg-0p1_standard","losg-1p0_standard",
                                                                      "log10_standard", "log10-sqrt_standard"])
        
    if verbose > 0:
        param_str = ' - '.join(['{}: {}'.format(k, v)
                                for k, v in trial.params.items()])
        print("Params: {}".format(param_str))
        
    transformer_type_dir = pl.Path("{}/{}".format(transformer_dir, transformer_type))

    train_batchset = load_batchset_sequence(array_dir=train_dir,
                                            meta_dir=train_dir,
                                            transformer_dir=transformer_type_dir,
                                            input_features=input_features,
                                            output_features=output_features,
                                            samples_size=samples_size,
                                            dates_size=dates_size,
                                            cuda = cuda,
                                            split_fraction=split_fraction,
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
                                                 transformer_dir=transformer_type_dir,
                                                 input_features=input_features,
                                                 output_features=output_features,
                                                 samples_size=samples_size,
                                                 dates_size=dates_size,
                                                 cuda=cuda,
                                                 split_fraction=split_fraction,
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
                                           transformer_dir=transformer_type_dir,
                                           input_features=input_features,
                                           output_features=output_features,
                                           samples_size=samples_size,
                                           dates_size=dates_size,
                                           cuda=cuda,
                                           split_fraction=split_fraction,
                                           seed=seed,
                                           verbose=verbose - 1)
    test_dataloader = load_dataloader_sequence(dataset=test_batchset,
                                               allow_prefetcher=not cuda,
                                               verbose=verbose - 1)
    
    model = SurrogateModel(input_size=input_features.size,
                           output_size=output_features.size,
                           n_lstm=n_lstm,
                           in_hidden_size=[in_hidden_size] * n_in_linear,
                           out_hidden_size=[out_hidden_size] * n_out_linear,
                           input_features=input_features,
                           output_features=output_features,
                           input_transformers=train_batchset.x_transformers,
                           output_transformers=train_batchset.y_transformers,
                           dropout_rate=dropout_rate,
                           try_cuda=True,
                           seed=seed)
    model.train(mode=False)
    if verbose > 0:
        print(model)

    optimizer = optim.AdamW(params=model.parameters(),
                            lr=learning_rate)
    
    trainer = ModuleTrainerSequence(train_loader=train_dataloader,
                                    validation_loader=validation_dataloader,
                                    test_loader=test_dataloader,
                                    model=model,
                                    optimizer=optimizer,
                                    transformers=list(test_batchset.y_transformers.values()))
    loss = nn.MSELoss(reduction = "mean")
    if verbose > 0:
        print(trainer)
        
    if verbose > 0:
        print("Runnning {} epochs".format(epochs))
    train_loss = trainer.run(epochs=epochs,
                             loss = loss,
                             seed=seed)
    if verbose > 0:
        print("Finished training with {}".format(train_loss))
    
    metrics = [loss]
    if test_metrics is not None:
        metrics = test_metrics
    test_losses = trainer.test(metrics = metrics)
    if verbose > 0:
        print("Finished testing with {}".format(test_losses))
        
    return test_losses
    
