from typing import Optional
import pandas as pd


def load_sequence_cells(ldd: pd.DataFrame,
                        subset_size: Optional[int] = None,
                        verbose: int = 1) -> tuple[list[list], pd.DataFrame]:

    upstream_from = set(ldd.index)
    upstream_to = set(ldd["downstream"])

    cell_indices = [cell
                    for cell in upstream_from
                    if cell not in upstream_to]
    #cell_indices.sort()

    ldd = ldd.drop(cell_indices)

    if verbose > 0:
        print("{} cells processed and {} upstream remaining".format(len(cell_indices),
                                                                    len(ldd.index)), flush=True)

    cell_indices_subset = [[index] for index in cell_indices]
    if subset_size is not None:
        indices_len = len(cell_indices)
        cell_indices_subset = [
            cell_indices[i:(i + subset_size)] for i in range(0, indices_len, subset_size)]

    return cell_indices_subset, ldd
