import pathlib as pl

import optuna as op
import pandas as pd
import optuna.storages as storages
import optuna.samplers as samplers
import optuna.pruners as pruners
import torch.nn as nn

from surrogate.run.tune.utils.optuna_objective import optuna_objective
from surrogate.nn.metrics import KGEMetric

setup_input_dir = pl.Path("../../setup/landsurface/input")
setup_dir = pl.Path("../../setup/landsurface/saves/train-test2")
transform_dir = pl.Path("../../transform/landsurface/saves/train-test2")
out_dir = pl.Path("./saves")
split_fraction = 1 / 32

seed = 19920223
epochs = 30
resolution = "mulres"

setup_input_resolution_dir = pl.Path("{}/{}".format(setup_input_dir, resolution))
setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
transform_resolution_dir = pl.Path("{}/{}".format(transform_dir, resolution))

in_features_file = pl.Path("{}/features_input.csv".format(setup_input_resolution_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features_file = pl.Path("{}/features_output.csv".format(setup_input_resolution_dir))
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

train_dir = pl.Path("{}/train".format(setup_resolution_dir))
validate_dir = pl.Path("{}/validate".format(setup_resolution_dir))
test_dir = pl.Path("{}/test".format(setup_resolution_dir))

database_file = pl.Path("{}/startup_database.db".format(out_dir))
database_file.parent.mkdir(parents=True, exist_ok=True)
storage = storages.RDBStorage(url="sqlite:///{}".format(database_file),
                                heartbeat_interval=600,
                                engine_kwargs={"connect_args": {"timeout": 1000}})

n_trials = 0
for study in storage.get_all_studies():
    n_trials += storage.get_n_trials(study._study_id)

sampler = samplers.RandomSampler(seed=seed + n_trials)
pruner = pruners.NopPruner()
study = op.create_study(directions=["minimize", "maximize"],
                        study_name="tune_parameters",
                        storage=storage,
                        load_if_exists=True,
                        sampler=sampler,
                        pruner=pruner)

print(study.trials_dataframe().head())

def optimize_trial(trial):            
    return optuna_objective(trial=trial,
                            epochs=epochs,
                            train_dir=train_dir,
                            validation_dir=validate_dir,
                            test_dir=test_dir,
                            transformer_dir=transform_resolution_dir,
                            input_features=in_features["feature"].to_numpy(),
                            output_features=out_features["feature"].to_numpy(),
                            test_metrics=[nn.MSELoss(reduction="mean"), KGEMetric(reduction="median")],
                            split_fraction=split_fraction,
                            seed=seed,
                            verbose=2)

study.optimize(optimize_trial, n_trials=1)