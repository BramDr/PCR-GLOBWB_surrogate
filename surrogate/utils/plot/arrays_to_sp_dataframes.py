from typing import Optional, Sequence

import pandas as pd
import numpy as np


def array_to_sp_dataframe(input: np.ndarray,
                          lats: np.ndarray,
                          lons: np.ndarray,
                          features: Optional[np.ndarray] = None,
                          save_variability: bool = False,
                          save_quantiles: bool = False,
                          save_range: bool = False,
                          spinup_len: int = 0,
                          verbose: int = 1) -> pd.DataFrame:

    if features is None:
        features_len = input.shape[-1]
        features = np.arange(features_len)

    input = input[spinup_len:, ...]

    if input.size == 0:
        input_df = pd.DataFrame(
            columns=["feature", "lat", "lon", "median", "mean"])
        if save_variability:
            input_df["std_variability"] = None
        if save_range:
            input_df["min_range"] = None
            input_df["max_range"] = None
        if save_quantiles:
            input_df["q1"] = None
            input_df["q3"] = None
        return input_df

    input_median = np.median(input, axis=0)
    input_median = input_median.flatten()
    input_mean = np.mean(input, axis=0)
    input_mean = input_mean.flatten()
    df_feature = np.tile(A=features, reps=lats.size)
    df_lats = np.repeat(a=lats, repeats=features.size)
    df_lons = np.repeat(a=lons, repeats=features.size)
    df_dict = {"feature": df_feature,
               "lat": df_lats,
               "lon": df_lons,
               "median": input_median,
               "mean": input_mean}

    input_df = pd.DataFrame(df_dict)

    if save_variability:
        input_std = np.std(input, axis=0)
        input_std = input_std.flatten()
        input_df["std_variability"] = input_std

    if save_range:
        input_min = np.min(input, axis=0)
        input_max = np.max(input, axis=0)
        input_min = input_min.flatten()
        input_max = input_max.flatten()
        input_df["min_range"] = input_min
        input_df["max_range"] = input_max
        
    if save_quantiles:
        input_q1 = np.quantile(input, q = 0.25, axis=0)
        input_q3 = np.quantile(input, q = 0.75, axis=0)
        input_q1 = input_q1.flatten()
        input_q3 = input_q3.flatten()
        input_df["q1"] = input_q1
        input_df["q3"] = input_q3

    if verbose > 0:
        input_df.head()

    return input_df


def arrays_to_sp_dataframes(inputs: Sequence[np.ndarray],
                            lats: np.ndarray,
                            lons: np.ndarray,
                            types: Optional[np.ndarray] = None,
                            features: Optional[np.ndarray] = None,
                            save_variability: bool = False,
                            save_quantiles: bool = False,
                            save_range: bool = False,
                            spinup_len: int = 0,
                            verbose: int = 1) -> pd.DataFrame:

    if types is None:
        types_len = len(inputs)
        types = np.arange(types_len)

    input_df = pd.DataFrame()
    for input, type in zip(inputs, types):
        if verbose > 0:
            print("Working on {} {}".format(type, input.shape), flush=True)

        df = array_to_sp_dataframe(input=input,
                                   features=features,
                                   lats=lats,
                                   lons=lons,
                                   save_variability=save_variability,
                                   save_quantiles=save_quantiles,
                                   save_range=save_range,
                                   spinup_len=spinup_len,
                                   verbose=verbose - 1)

        df.insert(0, column="type", value=type)
        input_df = pd.concat(objs=[input_df, df])

    return input_df
