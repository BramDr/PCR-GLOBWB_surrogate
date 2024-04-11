from typing import Optional
import datetime as dt
import plotnine as pn
import pandas as pd
import numpy as np


def plot_ts(input_df: pd.DataFrame,
            plot_mean: bool = True,
            plot_sensitivity: bool = False,
            sens_alpha_limit: tuple[float, float] = (0.1, 0.2),
            plot_range: bool = False,
            train_line: Optional[dt.datetime]=None,
            quantile_lim: bool = False,
            ylim: Optional[tuple] = None,
            ylab: Optional[str] = "Flux / State (mm)",
            xlab: Optional[str] = "Date") -> pn.ggplot:

    categories = ["setup", "transformed", "actual", "predicted", "30min", "05min", "multi-scale"]
    categories = [category for category in categories if category in pd.unique(input_df["type"])]
    input_df = input_df.assign(type = pd.Categorical(values=input_df["type"],
                                                     categories=categories))
    
    if ylim is None and quantile_lim:
        ylim_min = np.quantile(input_df["median"], 0.02)
        ylim_max = np.quantile(input_df["median"], 0.98)
        if plot_sensitivity:
            sens_len = len(sens_alpha_limit)
            for index in range(sens_len):
                min = np.quantile(input_df["min_sens_{}".format(index)], 0.25)
                max = np.quantile(input_df["max_sens_{}".format(index)], 0.98)
                if min < ylim_min:
                    ylim_min = min
                if max > ylim_max:
                    ylim_max = max
        if plot_range:
            min = np.quantile(input_df["min_range"], 0.02)
            max = np.quantile(input_df["max_range"], 0.98)
            if min < ylim_min:
                ylim_min = min
            if max > ylim_max:
                ylim_max = max
        if plot_mean:
            min = np.quantile(input_df["mean"], 0.02)
            max = np.quantile(input_df["mean"], 0.98)
            if min < ylim_min:
                ylim_min = min
            if max > ylim_max:
                ylim_max = max
        ylim = (ylim_min, ylim_max)
    
    plot = pn.ggplot(data=input_df,
                  mapping=pn.aes(x="date",
                              group="type",
                              color="type",
                              fill="type"))
    if plot_sensitivity:
        sens_len = len(sens_alpha_limit)
        sens_min = sens_alpha_limit[0]
        sens_max = sens_alpha_limit[1]
        alpha_diff = (sens_max - sens_min) / (sens_len - 1)
        for index in range(sens_len):
            alpha = sens_alpha_limit[0] + alpha_diff * index
            plot += pn.geom_ribbon(mapping=pn.aes(ymin="min_sens_{}".format(index),
                                                  ymax="max_sens_{}".format(index)),
                                alpha=alpha,
                                color="none")
    if plot_range:
        plot += pn.geom_line(mapping=pn.aes(y="min_range"),
                            linetype = "dotted")
        plot += pn.geom_line(mapping=pn.aes(y="max_range"),
                            linetype = "dotted")
    if plot_mean:
        plot += pn.geom_line(mapping=pn.aes(y="mean"),
                            linetype = "dashed",
                            alpha=0.98)
    plot += pn.geom_line(mapping=pn.aes(y="median"),
                         alpha=0.98)
    if train_line is not None:
        plot += pn.geom_vline(xintercept=train_line)
    plot += pn.facet_wrap(facets="~feature",
                       ncol=1,
                       scales="free_y")
    if xlab is not None:
        plot += pn.scale_x_datetime(name=xlab)
    if ylab is not None:
        plot += pn.scale_y_continuous(name=ylab)
    plot += pn.coord_cartesian(ylim=ylim)
    plot += pn.theme(panel_background=pn.element_blank(),
                  panel_grid_major=pn.element_blank(),
                  panel_grid_minor=pn.element_blank(),
                  figure_size=(8.4, 2.8))

    return plot


def plot_ts_seperate(input_df: pd.DataFrame,
                    plot_mean: bool = True,
                    plot_sensitivity: bool = False,
                    sens_alpha_limit: tuple[float, float] = (0.1, 0.2),
                    plot_range: bool = False,
                    train_line: Optional[dt.datetime]=None,
                    quantile_lim: bool = False,
                    ylim:  Optional[tuple]=None,
                    ylab: Optional[str] = "Flux / State (mm)",
                    xlab: Optional[str] = "Date") -> list[pn.ggplot]:
        
    features = input_df["feature"].unique()
        
    plots = []
    for feature in features:
        
        input_tmp = input_df.loc[input_df["feature"] == feature]
        plot = plot_ts(input_df = input_tmp,
                       plot_mean=plot_mean,
                       plot_sensitivity=plot_sensitivity,
                       sens_alpha_limit=sens_alpha_limit,
                       plot_range=plot_range,
                       train_line=train_line,
                       quantile_lim=quantile_lim,
                       ylim=ylim,
                       ylab=ylab,
                       xlab=xlab)
        plots.append(plot)
    
    return plots
        
        