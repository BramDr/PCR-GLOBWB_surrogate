import optuna as op
import pathlib as pl
import pathlib as pl
import pandas as pd

from utils.build_tune_model import build_tune_model

save_dir = pl.Path("../prepare/saves/global_30min")
transformer_dir = pl.Path("../transform/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min")
subset = "hyper"
seed = 19920232

n_backwards = 5000
n_params = 7
n_startup_trials = n_params * 25

upstream = pd.read_csv(upstream_file, index_col=0)

subset_dir = pl.Path("{}/{}".format(save_dir, subset))
transformer_subset_dir = pl.Path("{}/{}".format(transformer_dir, subset))

train_dir = pl.Path("{}/cells_training".format(subset_dir))
test_dirs = [pl.Path("{}/cells_validation_spatial".format(subset_dir)),
             pl.Path("{}/cells_validation_temporal".format(subset_dir)),
             pl.Path("{}/cells_validation_spatiotemporal".format(subset_dir))]

sampler = op.samplers.TPESampler()
pruner = op.pruners.HyperbandPruner(min_resource=30, reduction_factor=2)

database_file = pl.Path(
    "{}/distributed_optuna_study.db".format(dir_out))
database_file.parent.mkdir(parents=True,
                           exist_ok=True)
storage = op.storages.RDBStorage(url="sqlite:///{}".format(database_file),
                                 heartbeat_interval=60,
                                 engine_kwargs={"connect_args": {"timeout": 100}})

study = build_tune_model(n_trials=1,
                         train_dir=train_dir,
                         transformer_dir=transformer_subset_dir,
                         upstream=upstream,
                         n_backwards=n_backwards,
                         test_dirs=test_dirs,
                         storage=storage,
                         sampler=sampler,
                         pruner=pruner,
                         seed=seed,
                         verbose=2)

failed_trials = study.get_trials(deepcopy=False,
                                 states=[op.trial.TrialState.FAIL])
pruned_trials = study.get_trials(deepcopy=False,
                                 states=[op.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False,
                                   states=[op.trial.TrialState.COMPLETE])

print("Study statistics: ", flush=True)
print("  Number of finished trials: ", len(study.trials), flush=True)
print("  Number of failed trials: ", len(failed_trials), flush=True)
print("  Number of pruned trials: ", len(pruned_trials), flush=True)
print("  Number of complete trials: ", len(complete_trials), flush=True)

print("Best trial:", flush=True)
trial = study.best_trial

print("  Value: ", trial.value, flush=True)

print("  Params: ", flush=True)
for key, value in trial.params.items():
    print("    {}: {}".format(key, value), flush=True)
