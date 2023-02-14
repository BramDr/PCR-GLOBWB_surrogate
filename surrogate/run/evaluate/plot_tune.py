import pathlib as pl
import plotnine as pn
import numpy as np
import optuna as op
import optuna.visualization as vis
import optuna.importance as im

from utils.plot_studies_importance import plot_studies_importance

tune_dir = pl.Path("../tune/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/tune")

database_file = pl.Path("{}/startup/distributed_optuna_study.db".format(tune_dir))
storage = op.storages.RDBStorage(url="sqlite:///{}".format(database_file))
study_random = op.create_study(direction="maximize",
                                study_name="hyperparameter_optimization",
                                storage=storage,
                                load_if_exists=True)

storage = op.storages.InMemoryStorage()
study_random_clean = op.create_study(direction="maximize",
                                    study_name="hyperparameter_optimization",
                                    storage=storage)
for trial in study_random.trials:
    if trial.value == np.NINF:
        continue
    study_random_clean.add_trial(trial)
del study_random

database_file = pl.Path("{}/distributed_optuna_study.db".format(tune_dir))
storage = op.storages.RDBStorage(url="sqlite:///{}".format(database_file),
                                 heartbeat_interval=60,
                                 engine_kwargs={"connect_args": {"timeout": 100}})
study_optim = op.create_study(direction="maximize",
                                study_name="hyperparameter_optimization",
                                storage=storage,
                                load_if_exists=True)

storage = op.storages.InMemoryStorage()
study_optim_clean = op.create_study(direction="maximize",
                                    study_name="hyperparameter_optimization",
                                    storage=storage)
for trial in study_optim.trials:
    if trial.value == np.NINF:
        continue
    study_optim_clean.add_trial(trial)
del study_optim

plot = plot_studies_importance(studies=(study_optim_clean, study_random_clean),
                               subsets=("optimized", "random"))

plot_out = pl.Path("{}/param_importance.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
pn.save_as_pdf_pages(plots=[plot], filename=plot_out)

vis.plot_contour(study=study_optim_clean, params=["learning_rate", "dropout_rate"])
vis.plot_contour(study=study_optim_clean, params=["dates_size", "sample_size"])

print(study_optim_clean.best_params)