from typing import Optional

import pandas as pd
import numpy as np


def load_sequential_cells_reverse(cells: pd.DataFrame,
                                  ldd: pd.DataFrame,
                                  subset_size: Optional[int] = None,
                                  max_tries: int = 1000,
                                  verbose: int = 1) -> list[list[list]]:
    
    cell_indices_sequential = []
    
    upstream_sel = ~np.isin(cells.index, ldd.index)
    cell_indices = cells.index[upstream_sel]
    
    cell_indices_subset = [[index] for index in cell_indices]
    if subset_size is not None:
        indices_len = len(cell_indices)
        cell_indices_subset = [
            cell_indices[i:(i + subset_size)] for i in range(0, indices_len, subset_size)]
    cell_indices_sequential.append(cell_indices_subset)
    
    iteration = 0
    for iteration in range(max_tries):
        if verbose > 0:
            print("Working on iteration {}".format(iteration), flush=True)
            
        upstream_sel = np.isin(ldd["downstream"], cell_indices)
        cell_indices = ldd.index[upstream_sel]
        ldd = ldd.loc[~upstream_sel]
        
        if verbose > 0:
            print("{} cells processed and {} upstream remaining".format(len(cell_indices),
                                                                        len(ldd.index)), flush=True)
        
        cell_indices_subset = [[index] for index in cell_indices]
        if subset_size is not None:
            indices_len = len(cell_indices)
            cell_indices_subset = [
                cell_indices[i:(i + subset_size)] for i in range(0, indices_len, subset_size)]
        cell_indices_sequential.append(cell_indices_subset)
        
        if len(ldd.index) == 0:
            break
    
    cell_indices_sequential.reverse()
    
    return cell_indices_sequential
