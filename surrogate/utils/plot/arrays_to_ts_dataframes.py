from typing import Optional, Sequence

import pandas as pd
import numpy as np


def array_to_ts_dataframe(input: np.ndarray,
                          features: Optional[Sequence] = None,
                          sequences: Optional[Sequence] = None,
                          save_sensitivity: bool = False,
                          sens_limits: Sequence[tuple[float, float]] = [(0.15, 0.85), (0.33, 0.66)],
                          save_range: bool = False,
                          verbose: int = 1) -> pd.DataFrame:
    
    if features is None:
        features_len = input.shape[2]
        features = [f for f in range(features_len)]
    
    if sequences is None:
        sequences_len = input.shape[1]
        sequences = [s for s in range(sequences_len)]
    
    if input.size == 0:
        input_df = pd.DataFrame(columns=["feature", "date", "median", "mean"])
        if save_sensitivity:
            for index, sens_limit in enumerate(sens_limits):
                input_df["min_sens_{}".format(index)] = None
                input_df["max_sens_{}".format(index)] = None
        if save_range:
            input_df["min_range"] = None
            input_df["max_range"] = None
        return input_df
    
    input_median = np.median(input, axis=0)
    input_median = input_median.flatten()
    input_mean = np.mean(input, axis=0)
    input_mean = input_mean.flatten()
    df_feature = [f for _ in sequences for f in features]
    df_sequence = [s for s in sequences for _ in features]
    df_dict = {"feature": df_feature,
               "date": df_sequence,
               "median": input_median,
               "mean": input_mean}

    input_df = pd.DataFrame(df_dict)

    if save_sensitivity:
        for index, sens_limit in enumerate(sens_limits):
            input_min = np.quantile(input, q=sens_limit[0], axis=0)
            input_max = np.quantile(input, q=sens_limit[1], axis=0)
            input_min = input_min.flatten()
            input_max = input_max.flatten()
            input_df["min_sens_{}".format(index)] = input_min
            input_df["max_sens_{}".format(index)] = input_max

    if save_range:
        input_min = np.min(input, axis=0)
        input_max = np.max(input, axis=0)
        input_min = input_min.flatten()
        input_max = input_max.flatten()
        input_df["min_range"] = input_min
        input_df["max_range"] = input_max

    if verbose > 0:
        input_df.head()

    return input_df


def arrays_to_ts_dataframes(inputs: Sequence[np.ndarray],
                            types: Optional[Sequence] = None,
                            features: Optional[Sequence] = None,
                            sequences: Optional[Sequence] = None,
                            save_sensitivity: bool = False,
                            sens_limits: Sequence[tuple[float, float]] = [(0.15, 0.85), (0.33, 0.66)],
                            save_range: bool = False,
                            verbose: int = 1) -> pd.DataFrame:

    if types is None:
        types_len = len(inputs)
        types = [t for t in range(types_len)]
        
    input_df = pd.DataFrame()
    for input, type in zip(inputs, types):
        if verbose > 0:
            print("Working on {} {}".format(type, input.shape), flush = True)

        df = array_to_ts_dataframe(input=input,
                                   features=features,
                                   sequences=sequences,
                                   save_sensitivity=save_sensitivity,
                                   sens_limits=sens_limits,
                                   save_range=save_range,
                                   verbose=verbose - 1)

        df.insert(0, column="type", value=type)
        input_df = pd.concat(objs=[input_df, df])

    return input_df
