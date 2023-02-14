from typing import Optional
import datetime as dt
import plotnine as pn
import pandas as pd
import patchworklib as pwl

from .plot_ts import plot_ts

def diff_ts(input_df: pd.DataFrame) -> pd.DataFrame:
    sens_keys = [key for key in input_df.keys() if "sens_" in key]
    sens_indices = [int(key.split("_")[2]) for key in sens_keys]
    sens_len = max(sens_indices)
    
    types = input_df["type"].unique()
    if len(types) != 2:
        print("Spatial difference dataframe should have exacty two types, returning empty plot")
        return pd.DataFrame()
    
    type_left = types[0]
    type_right = types[1]
    input_df_left = input_df.loc[input_df["type"] == type_left]
    input_df_right = input_df.loc[input_df["type"] == type_right]
        
    metric_keys = [key for key in input_df.keys() if key in ["type", "mean", "median", "min_range", "max_range"]]
    for index in range(sens_len):
        metric_keys += ["min_sens_{}".format(index), "max_sens_{}".format(index)]
    merge_keys = [key for key in input_df.keys() if key not in metric_keys]
            
    input_df_diff = pd.merge(left = input_df_left,
                             right = input_df_right,
                             on = merge_keys)    
    input_df_diff = input_df_diff.reset_index(drop=True)
    
    for key in metric_keys:
        if key == "type":
            continue
        
        key_x = "{}_x".format(key)
        key_y = "{}_y".format(key)
        if key_x not in input_df_diff.keys() or key_y not in input_df_diff.keys():
            continue
        
        input_df_diff[key] = input_df_diff[key_x] - input_df_diff[key_y]
        input_df_diff = input_df_diff.drop([key_x, key_y], axis=1)
    
    type_diff = "Difference ({} - {})".format(type_left, type_right)
    input_df_diff["type"] = type_diff
    
    return input_df_diff


def plot_ts_diff_seperate(input_df: pd.DataFrame,
                    plot_sensitivity: bool = False,
                    sens_alpha_limit: tuple[float, float] = (0.1, 0.2),
                    plot_range: bool = False,
                    train_line: Optional[dt.datetime]=None,
                    ylim:  Optional[tuple]=(None, None),
                    ylab: Optional[str] = "Flux / State (mm)",
                    xlab: Optional[str] = "Date") -> list[pn.ggplot]:
    
    input_df_diff = diff_ts(input_df=input_df)
    
    features = input_df_diff["feature"].unique()
        
    plots = []
    for feature in features:
        print("Processing {}".format(feature))
        input_df_feature = input_df.loc[input_df["feature"] == feature]
        input_df_diff_feature = input_df_diff.loc[input_df_diff["feature"] == feature]
        
        plot_base = plot_ts(input_df = input_df_feature,
                            plot_sensitivity=plot_sensitivity,
                            sens_alpha_limit=sens_alpha_limit,
                            plot_range=plot_range,
                            train_line=train_line,
                            ylim=ylim,
                            ylab=ylab,
                            xlab=xlab)
        
        plot_diff = plot_ts(input_df = input_df_diff_feature,
                            plot_sensitivity=plot_sensitivity,
                            sens_alpha_limit=sens_alpha_limit,
                            plot_range=plot_range,
                            train_line=train_line,
                            ylim=(-0.5, 0.5),
                            ylab=ylab,
                            xlab=xlab)
        
        
        old_version = pn.__version__
        pn.__version__ = "0.9.0"
        plot_base = pwl.load_ggplot(plot_base)
        plot_diff = pwl.load_ggplot(plot_diff)
        pn.__version__ = old_version
        
        plot = (plot_base | plot_diff)
        plots.append(plot)
    
    return plots
        
        