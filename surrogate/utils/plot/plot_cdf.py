from typing import Optional
import plotnine as pn
import pandas as pd


def plot_cdf(input_df: pd.DataFrame,
             ylab: Optional[str] = "Cumulative probability",
             xlab: Optional[str] = "x (mm)") -> pn.ggplot:

    plot = pn.ggplot(data=input_df,
                  mapping=pn.aes(x="value",
                                 y="prob",
                                 group="type",
                                 color="type"))
    plot += pn.geom_line()
    plot += pn.facet_wrap(facets="~feature",
                       ncol=1,
                       scales="free_y")
    if xlab is not None:
        plot += pn.scale_x_continuous(name=xlab)
    if ylab is not None:
        plot += pn.scale_y_continuous(name=ylab)
    plot += pn.theme(panel_background=pn.element_blank(),
                  panel_grid_major=pn.element_blank(),
                  panel_grid_minor=pn.element_blank())

    return plot


def plot_cdf_seperate(input_df: pd.DataFrame,
                        ylab: Optional[str] = "Cumulative probability",
                        xlab: Optional[str] = "x (mm)") -> list[pn.ggplot]:
        
    features = input_df["feature"].unique()
    
    plots = []
    for feature in features:
        print("Processing {}".format(feature))
        input_tmp = input_df.loc[input_df["feature"] == feature]
        plot = plot_cdf(input_df=input_tmp,
                            ylab=ylab,
                            xlab=xlab)
        plots.append(plot)
    
    return plots
        
        