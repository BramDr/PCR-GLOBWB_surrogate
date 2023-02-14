from typing import Optional
import plotnine as pn
import pandas as pd
import patchworklib as pwl

from .plot_sp import plot_sp

def diff_sp(input_df: pd.DataFrame) -> pd.DataFrame:
    
    types = input_df["type"].unique()
    if len(types) != 2:
        print("Spatial difference dataframe should have exacty two types, returning empty plot")
        return pd.DataFrame()
    
    type_left = types[0]
    type_right = types[1]
    input_df_left = input_df.loc[input_df["type"] == type_left]
    input_df_right = input_df.loc[input_df["type"] == type_right]
        
    metric_keys = [key for key in input_df.keys() if key in ["type", "value"]]
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


def plot_sp_diff_seperate(input_df: pd.DataFrame,
                          statistics: list[str] = ["mean", "std_variability"],
                          filllim:  Optional[tuple]=(None, None),
                          xlab: Optional[str] = "longitude (degrees east)",
                          ylab: Optional[str] = "latitude (degrees north)") -> list[pn.ggplot]:
        
    input_df = pd.melt(frame = input_df,
                       id_vars = ["feature", "type", "lat", "lon"],
                       value_vars = statistics,
                       var_name = "statistic",
                       value_name="value")
    
    input_df_diff = diff_sp(input_df=input_df)
    
    features = input_df_diff["feature"].unique()
    
    plots = []
    for feature in features:
        print("Processing {}".format(feature))
        input_df_feature = input_df.loc[input_df["feature"] == feature]
        input_df_diff_feature = input_df_diff.loc[input_df_diff["feature"] == feature]
        
        plot_base = plot_sp(input_df=input_df_feature,
                            filllim=filllim,
                            ylab=ylab,
                            xlab=xlab)
        
        plot_diff = plot_sp(input_df=input_df_diff_feature,
                            filllim=(-0.5, 0.5),
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
            
            