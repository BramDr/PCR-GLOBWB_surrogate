import optuna.trial
import optuna.exceptions

from .Callback import Callback


class OptunaReporter(Callback):
    def __init__(self,
                 trial: optuna.trial.Trial) -> None:
        super().__init__()

        self.trial = trial

    def epoch_end_callback(self, trainer, epoch: int, statistics: dict) -> None:
        try:
            loss = statistics["test_loss"]
        except KeyError:
            loss = statistics["train_loss"]

        self.trial.report(value=-loss, step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
