import csv
import typing
import pathlib as pl

from .Logger import Logger


class CsvDictLogger(Logger):
    def __init__(self,
                 path: pl.Path) -> None:
        super(CsvDictLogger, self).__init__(path=path)

    def write_dict(self, dictionaries: typing.Iterable[dict]) -> None:
        super(CsvDictLogger, self).write()

        with open(self.path, 'a', newline='') as file:
            dict_object = csv.DictWriter(
                file, fieldnames=dictionaries[0].keys())

            if self.first:
                dict_object.writeheader()
                self.first = False

            for dictionary in dictionaries:
                dict_object.writerow(dictionary)
