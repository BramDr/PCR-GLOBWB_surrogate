from typing import Optional, Sequence
import pathlib as pl
import pandas as pd
import optuna as op

import torch.utils.data as data

from surrogate.utils.train import OptunaReporter
from surrogate.utils.data import load_batchsets
from surrogate.utils.data import load_batchsets
from surrogate.utils.data import load_dataloader
from surrogate.utils.data import load_dataloaders
from surrogate.nn import build_model
from surrogate.utils.train import build_trainer

def optuna_model_objective(trial: op.trial.Trial,
                            train_dir: pl.Path,
                            transformer_dir: pl.Path,
                            upstream: pd.DataFrame,
                            n_backwards: int,
                            test_dirs: Optional[Sequence[pl.Path]] = None,
                            cuda: bool = True,
                            allow_prefetcher: bool = True,
                            seed: int = 19920223,
                            verbose: int = 1) -> float:
    
    if test_dirs is None:
        test_dirs = []
    
    n_lstm = trial.suggest_int("n_lstm", 1, 2)
    n_linear = trial.suggest_int("n_linear", 1, 2)
    hidden_size = trial.suggest_int("hidden_size", 16, 512)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.75)
    sample_size = trial.suggest_int("sample_size", 1, 64)
    dates_size = trial.suggest_int("dates_size", 50, 1000)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    
    if verbose > 0:
        param_str = ' - '.join(['{}: {}'.format(k, v)
                                for k, v in trial.params.items()])
        print("Params: {}".format(param_str), flush=True)

    train_dataset = load_batchsets(save_dir=train_dir,
                                transformer_dir=transformer_dir,
                                upstream=upstream,
                                cuda=True,
                                sample_size=sample_size,
                                dates_size=dates_size,
                                verbose=verbose - 1)
    
    test_datasets = load_batchsets(save_dirs=test_dirs,
                                transformer_dir=transformer_dir,
                                upstream=upstream,
                                cuda=True,
                                verbose=verbose)
    
    train_sampler = data.RandomSampler(data_source=train_dataset)
    train_dataloader = load_dataloader(dataset=train_dataset,
                                       sampler=train_sampler,
                                       allow_prefetcher=cuda and allow_prefetcher,
                                       verbose=verbose - 1)

    test_samplers = []
    for dataset in test_datasets:
        sampler = data.SequentialSampler(data_source=dataset)
        test_samplers.append(sampler)
    test_dataloaders = load_dataloaders(datasets=test_datasets,
                                        samplers=test_samplers,
                                        allow_prefetcher=cuda and allow_prefetcher,
                                        verbose=verbose)
    
    input_size = train_dataset.in_features_len
    output_size = train_dataset.out_features_len
    in_hidden_size = [hidden_size] * n_linear
    out_hidden_size = [hidden_size] * n_linear
    model = build_model(input_size=input_size,
                        output_size=output_size,
                        n_lstm=n_lstm,
                        in_hidden_size=in_hidden_size,
                        out_hidden_size=out_hidden_size,
                        dropout_rate=dropout_rate,
                        try_cuda=cuda,
                        seed=seed,
                        verbose=verbose)
    
    callback = OptunaReporter(trial=trial)
    callbacks = [callback]

    trainer = build_trainer(dataloader=train_dataloader,
                            model=model,
                            test_dataloaders=test_dataloaders,
                            learning_rate=learning_rate,
                            #weight_decay=weight_decay,
                            callbacks=callbacks,
                            verbose=verbose)

    epochs = int(n_backwards / len(train_dataset))
    if verbose > 0:
        print("Runnning {} epochs".format(epochs), flush=True)
        
    best_loss = trainer.train(epochs=epochs,
                              seed=seed,
                              verbose=verbose)
    
    if verbose > 0:
        print("Finished training with {}".format(best_loss), flush=True)
        
    return -best_loss
    
