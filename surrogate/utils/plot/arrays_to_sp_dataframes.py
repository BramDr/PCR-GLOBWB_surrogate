from typing import Optional, Sequence

import pandas as pd
import numpy as np


def array_to_sp_dataframe(input: np.ndarray,
                          lats: Sequence,
                        lons: Sequence,
                        features: Optional[Sequence] = None,
                          save_variability: bool = False,
                          save_range: bool = False,
                          verbose: int = 1) -> pd.DataFrame:
    
    if features is None:
        features_len = input.shape[2]
        features = [f for f in range(features_len)]
    
    if input.size == 0:
        input_df = pd.DataFrame(columns=["feature", "lat", "lon", "median", "mean"])
        if save_variability:
            input_df["std_variability"] = None
        if save_range:
            input_df["min_range"] = None
            input_df["max_range"] = None
        return input_df
    
    input_median = np.median(input, axis=1)
    input_median = input_median.flatten()
    input_mean = np.mean(input, axis=1)
    input_mean = input_mean.flatten()    
    df_feature = [f for _ in lats for f in features]
    df_lats = [l for l in lats for _ in features]
    df_lons = [l for l in lons for _ in features]    
    df_dict = {"feature": df_feature,
               "lat": df_lats,
               "lon": df_lons,
               "median": input_median,
               "mean": input_mean}

    input_df = pd.DataFrame(df_dict)

    if save_variability:
        input_std = np.std(input, axis=1)
        input_std = input_std.flatten()
        input_df["std_variability"] = input_std

    if save_range:
        input_min = np.min(input, axis=1)
        input_max = np.max(input, axis=1)
        input_min = input_min.flatten()
        input_max = input_max.flatten()
        input_df["min_range"] = input_min
        input_df["max_range"] = input_max

    if verbose > 0:
        input_df.head()

    return input_df


def arrays_to_sp_dataframes(inputs: list[np.ndarray],
                            lats: Sequence,
                            lons: Sequence,
                            types: Optional[Sequence] = None,
                            features: Optional[Sequence] = None,
                            save_variability: bool = False,
                            save_range: bool = False,
                            verbose: int = 1) -> pd.DataFrame:

    if types is None:
        types_len = len(inputs)
        types = [t for t in range(types_len)]

    input_df = pd.DataFrame()
    for input, type in zip(inputs, types):
        if verbose > 0:
            print("Working on {} {}".format(type, input.shape), flush = True)

        df = array_to_sp_dataframe(input=input,
                                   features=features,
                                   lats=lats,
                                   lons=lons,
                                   save_variability=save_variability,
                                   save_range=save_range,
                                   verbose=verbose - 1)

        df.insert(0, column="type", value=type)
        input_df = pd.concat(objs=[input_df, df])

    return input_df
