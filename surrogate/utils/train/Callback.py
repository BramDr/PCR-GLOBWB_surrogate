import abc

class Callback(abc.ABC):
    def __init__(self) -> None:
        pass

    def start_callback(self,
                       trainer) -> None:
        pass

    def epoch_end_callback(self,
                           trainer,
                           epoch: int,
                           statistics: dict) -> None:
        pass

    def end_callback(self,
                     trainer) -> None:
        pass
