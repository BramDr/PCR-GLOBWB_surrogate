import os
import pathlib as pl


class Logger():
    def __init__(self,
                 path: pl.Path) -> None:
        super(Logger, self).__init__()

        self.first = True
        self.path = path

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.first and self.path.exists():
            os.remove(self.path)
