import pathlib as pl

import optuna as op
import pandas as pd
import optuna.storages as storages
import optuna.samplers as samplers
import optuna.pruners as pruners
import torch.nn as nn

from surrogate.run.tune.utils.optuna_objective import optuna_objective
from surrogate.nn.metrics import KGEAlphaMetric

setup_input_dir = pl.Path("../../setup/routing/input")
setup_dir = pl.Path("../../setup/routing/saves/train-test2")
transform_dir = pl.Path("../../transform/routing/saves/train-test2")
out_dir = pl.Path("./saves")
routing_split_fractions = {"river": 1 / 32,
                           "reservoir": 1 / 8,
                           "lake": 1 / 8,}

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

routing_types = [dir.stem for dir in setup_resolution_dir.iterdir() if dir.is_dir()]

routing_type = routing_types[0]
for routing_type in routing_types:
    print("Routing type: {}".format(routing_type))
    
    setup_routing_dir = pl.Path("{}/{}".format(setup_resolution_dir, routing_type))
    transform_routing_dir = pl.Path("{}/{}".format(transform_resolution_dir, routing_type))
    out_routing_dir = pl.Path("{}/{}".format(out_dir, routing_type))
    
    split_fraction = routing_split_fractions[routing_type]
    
    train_dir = pl.Path("{}/train".format(setup_routing_dir))
    validate_dir = pl.Path("{}/validate".format(setup_routing_dir))
    test_dir = pl.Path("{}/test".format(setup_routing_dir))
    
    database_file = pl.Path("{}/startup_database.db".format(out_routing_dir))
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
                                transformer_dir=transform_routing_dir,
                                input_features=in_features["feature"].to_numpy(),
                                output_features=out_features["feature"].to_numpy(),
                                test_metrics=[nn.MSELoss(reduction="mean"), KGEAlphaMetric(reduction="median")],
                                split_fraction=split_fraction,
                                seed=seed,
                                verbose=2)

    study.optimize(optimize_trial, n_trials=1)
