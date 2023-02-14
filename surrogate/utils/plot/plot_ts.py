from typing import Optional
import datetime as dt
import plotnine as pn
import pandas as pd


def plot_ts(input_df: pd.DataFrame,
            plot_sensitivity: bool = False,
            sens_alpha_limit: tuple[float, float] = (0.1, 0.2),
            plot_range: bool = False,
            train_line: Optional[dt.datetime]=None,
            ylim: Optional[tuple] = (None, None),
            ylab: Optional[str] = "Flux / State (mm)",
            xlab: Optional[str] = "Date") -> pn.ggplot:

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
                                color=(0, 0, 0, 0))
    if plot_range:
        plot += pn.geom_line(mapping=pn.aes(y="min_range"),
                            linetype = "dotted")
        plot += pn.geom_line(mapping=pn.aes(y="max_range"),
                            linetype = "dotted")
    plot += pn.geom_line(mapping=pn.aes(y="mean"),
                         linetype = "dashed",
                         alpha=0.33)
    plot += pn.geom_line(mapping=pn.aes(y="median"),
                         alpha=0.33)
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
                  panel_grid_minor=pn.element_blank())

    return plot


def plot_ts_seperate(input_df: pd.DataFrame,
                    plot_sensitivity: bool = False,
                    sens_alpha_limit: tuple[float, float] = (0.1, 0.2),
                    plot_range: bool = False,
                    train_line: Optional[dt.datetime]=None,
                    ylim:  Optional[tuple]=(None, None),
                    ylab: Optional[str] = "Flux / State (mm)",
                    xlab: Optional[str] = "Date") -> list[pn.ggplot]:
        
    features = input_df["feature"].unique()
        
    plots = []
    for feature in features:
        print("Processing {}".format(feature))
        input_tmp = input_df.loc[input_df["feature"] == feature]
        plot = plot_ts(input_df = input_tmp,
                       plot_sensitivity=plot_sensitivity,
                       sens_alpha_limit=sens_alpha_limit,
                       plot_range=plot_range,
                       train_line=train_line,
                       ylim=ylim,
                       ylab=ylab,
                       xlab=xlab)
        plots.append(plot)
    
    return plots
        
        