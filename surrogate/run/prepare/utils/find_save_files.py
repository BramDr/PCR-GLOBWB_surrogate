from typing import Optional
import pathlib as pl
import re


def _has_patterns(file: str,
                  patterns: list[str]) -> bool:
    for pattern in patterns:
        if re.match(pattern=pattern, string=file):
            return True
    return False


def find_save_files(dir: pl.Path,
                    include_patterns: Optional[list[str]] = None,
                    exclude_patterns: Optional[list[str]] = None,
                    verbose: int = 1) -> tuple[list[pl.Path], list[pl.Path]]:

    array_files = [file for file in dir.rglob("*.npy")]

    if include_patterns is not None:

        new_array_files = []
        for array_file in array_files:
            if _has_patterns(file=str(array_file),
                             patterns=include_patterns):
                new_array_files.append(array_file)
        array_files = new_array_files

    if exclude_patterns is not None:

        new_array_files = []
        for array_file in array_files:
            if not _has_patterns(file=str(array_file),
                                 patterns=exclude_patterns):
                new_array_files.append(array_file)
        array_files = new_array_files

    meta_files = [pl.Path("{}/{}_meta.pkl".format(file.parent, file.stem)) for file in array_files]

    array_files = [file for file in array_files]
    meta_files = [file for file in meta_files]

    if verbose > 0:
        print("Found {} array files and {} meta files".format(len(array_files),
                                                              len(meta_files)), flush=True)

    return array_files, meta_files
