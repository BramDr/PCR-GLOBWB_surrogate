from .Callback import Callback


class BestLoss(Callback):
    def __init__(self) -> None:
        super().__init__()

        self.best_loss = float('inf')

    def epoch_end_callback(self, trainer, epoch: int, statistics: dict) -> None:
        self.assess_loss(statistics=statistics)

    def assess_loss(self, statistics: dict) -> bool:
        try:
            loss = statistics["test_loss"]
        except KeyError:
            loss = statistics["train_loss"]

        if loss > self.best_loss:
            return False

        self.best_loss = loss
        return True
