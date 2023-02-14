from typing import Optional, Callable
import pandas as pd
import numpy as np


def array_to_perf_dataframe(true: np.ndarray,
                            pred: np.ndarray,
                            performance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            cells: pd.DataFrame,
                            features: np.ndarray,
                            train_cells: Optional[pd.DataFrame] = None,
                            verbose: int = 1) -> pd.DataFrame:

    performance = performance_fn(true, pred)
    n_sample = performance.shape[0]

    performance = performance.flatten()
    df_feature = [f for _ in range(n_sample) for f in features]
    df_cell = [cells.index[s] for s in range(n_sample) for _ in features]
    df_dict = {"feature": df_feature,
               "cell": df_cell,
               "value": performance}
    perf_df = pd.DataFrame(df_dict)

    perf_df["lon"] = cells["lon"].loc[perf_df["cell"].to_list()
                                      ].to_list()
    perf_df["lat"] = cells["lat"].loc[perf_df["cell"].to_list()
                                      ].to_list()

    perf_df["train"] = False
    if train_cells is not None:
        train_sel = [cell in train_cells.index for cell in perf_df["cell"]]
        perf_df.loc[train_sel, "train"] = True

    if verbose > 0:
        perf_df.head()

    return perf_df
