from typing import Optional, Sequence
import pandas as pd
import numpy as np


def _calc_probs(input: np.ndarray) -> np.ndarray:
    probs = 1. * np.arange(input.size) / (input.size - 1)
    return probs


def array_to_cdf_dataframe(input: np.ndarray,
                           features: Optional[Sequence] = None,
                           max_values: Optional[int] = None,
                           verbose: int = 1) -> pd.DataFrame:
    
    if features is None:
        features_len = input.shape[2]
        features = [f for f in range(features_len)]
        
    if input.size == 0:
        input_df = pd.DataFrame(columns=["feature", "prob", "value"])
        return input_df
    
    input = np.reshape(input, (-1, input.shape[2]))

    input = np.sort(input, axis=0)
    prob = np.apply_along_axis(_calc_probs, axis=0, arr=input)

    if max_values is not None:
        agg_factor = int(input.shape[0] / max_values)
        input = input[0:-1:agg_factor, :]
        prob = prob[0:-1:agg_factor, :]

    df_feature = [f for _ in range(input.shape[0]) for f in features]
    df_dict = {"feature": df_feature,
               "prob": prob.flatten(),
               "value": input.flatten()}

    input_df = pd.DataFrame(df_dict)

    if verbose > 0:
        input_df.head()

    return input_df


def arrays_to_cdf_dataframes(inputs: list[np.ndarray],
                             types: Optional[Sequence] = None,
                             features: Optional[Sequence] = None,
                             max_values: Optional[int] = None,
                             verbose: int = 1) -> pd.DataFrame:

    if types is None:
        types_len = len(inputs)
        types = [t for t in range(types_len)]
        
    input_df = pd.DataFrame()
    for input, type in zip(inputs, types):
        if verbose > 0:
            print("Working on {} {}".format(type, input.shape), flush = True)

        df_tmp = array_to_cdf_dataframe(input=input,
                                        features=features,
                                        max_values=max_values,
                                        verbose=verbose - 1)

        df_tmp.insert(0, column="type", value=type)
        input_df = pd.concat(objs=[input_df, df_tmp])

    return input_df
