from typing import Optional
import pathlib as pl
import pandas as pd
import optuna as op

from .optuna_model_objective import optuna_model_objective

def build_tune_model(n_trials: int,
               train_dir: pl.Path,
                transformer_dir: pl.Path,
                upstream: pd.DataFrame,
                n_backwards: int,
                test_dirs: Optional[list[pl.Path]] = [],
                study_name: str = "hyperparameter_optimization",
                storage: Optional[op.storages.BaseStorage] = None,
                sampler: Optional[op.samplers.BaseSampler] = None,
                pruner: Optional[op.pruners.BasePruner] = None,
                cuda: bool = True,
                allow_prefetcher: bool = True,
                seed: int = 19920223,
                verbose: int = 1) -> op.Study:

    study = op.create_study(direction="maximize",
                            study_name=study_name,
                            storage=storage,
                            load_if_exists=True,
                            sampler=sampler,
                            pruner=pruner)
    op.storages.fail_stale_trials(study=study)
    
    if verbose > 0:
        print(study.trials_dataframe(), flush=True)

    def optimize_trial(trial):
        return optuna_model_objective(trial = trial,
                                      train_dir=train_dir,
                                      transformer_dir = transformer_dir,
                                      upstream=upstream,
                                      n_backwards=n_backwards,
                                      test_dirs = test_dirs,
                                      cuda=cuda,
                                      allow_prefetcher=allow_prefetcher,
                                      seed=seed,
                                      verbose=verbose)
        
    study.optimize(optimize_trial, n_trials=n_trials)
    
    return study
