from typing import Optional
import pandas as pd

from .load_sequence_cells import load_sequence_cells


def load_sequential_cells(cells: pd.DataFrame,
                          ldd: pd.DataFrame,
                          subset_size: Optional[int] = None,
                          max_tries: int = 1000,
                          verbose: int = 1) -> list[list[list]]:
    
    upstream_from = set(ldd.index)
    upstream_to = set(ldd["downstream"])

    cell_indices_sequential = []
    
    cell_indices = [cell
                    for cell in cells.index
                    if cell not in upstream_to and cell not in upstream_from]
    #cell_indices.sort()

    cell_indices_subset = [[index] for index in cell_indices]
    if subset_size is not None:
        indices_len = len(cell_indices)
        cell_indices_subset = [
            cell_indices[i:(i + subset_size)] for i in range(0, indices_len, subset_size)]

    cell_indices_sequential.append(cell_indices_subset)
    
    for sequence in range(max_tries):
        if verbose > 0:
            print("Working on sequence {}".format(sequence), flush=True)

        cell_indices_subset, ldd = load_sequence_cells(ldd=ldd,
                                                       subset_size=subset_size,
                                                       verbose=verbose - 1)
        cell_indices_sequential.append(cell_indices_subset)

        if len(ldd.index) == 0:
            break

    sequential_len = len(cell_indices_sequential)
    if sequential_len >= max_tries:
        raise ValueError(
            "Maximum tries of {} is insufficient".format(max_tries))
    
    cell_indices = [cell
                    for cell in cells.index
                    if cell in upstream_to and cell not in upstream_from]
    #cell_indices.sort()

    cell_indices_subset = [[index] for index in cell_indices]
    if subset_size is not None:
        indices_len = len(cell_indices)
        cell_indices_subset = [
            cell_indices[i:(i + subset_size)] for i in range(0, indices_len, subset_size)]
        
    cell_indices_sequential.append(cell_indices_subset)
    
    return cell_indices_sequential
