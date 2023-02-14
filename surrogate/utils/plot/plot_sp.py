from typing import Optional
import plotnine as pn
import pandas as pd

def plot_sp(input_df: pd.DataFrame,
            filllim:  Optional[tuple]=(None, None),
            xlab: Optional[str] = "longitude (degrees east)",
            ylab: Optional[str] = "latitude (degrees north)"):
    
    input_df = input_df.sort_values(by=["lon", "lat"])
    
    plot = pn.ggplot(mapping=pn.aes(x="lon",
                                    y="lat",
                                    fill="value"))
    plot += pn.geom_raster(data = input_df,
                           raster = False,
                           interpolation="none")
    plot += pn.scale_fill_distiller(name="statistic",
                                    type="div",
                                    palette="RdYlBu",
                                    limits = filllim,
                                    direction=-1)
    plot += pn.coord_fixed()
    plot += pn.facet_grid(facets="statistic~type+feature",
                          scales="free_fill")
    plot += pn.scale_x_continuous(name=xlab)
    plot += pn.scale_y_continuous(name=ylab)
    plot += pn.theme(panel_background=pn.element_blank(),
                    panel_grid_major=pn.element_blank(),
                    panel_grid_minor=pn.element_blank())

    return plot


def plot_sp_seperate(input_df: pd.DataFrame,
                     statistics: list[str] = ["mean", "std_variability"],
                     filllim:  Optional[tuple]=(None, None),
                     xlab: Optional[str] = "longitude (degrees east)",
                     ylab: Optional[str] = "latitude (degrees north)") -> list[pn.ggplot]:
    
    if "type" not in input_df.keys():
        input_df["type"] = "input"
    
    features = input_df["feature"].unique()
    
    plots = []
    for feature in features:
        print("Processing {}".format(feature))
        
        input_feature = input_df.loc[input_df["feature"] == feature]
        input_feature = input_feature.reset_index(drop = True)
        
        input_feature = pd.melt(frame = input_feature,
                            id_vars = ["feature", "type", "lat", "lon"],
                            value_vars = statistics,
                            var_name = "statistic",
                            value_name="value")
        
        plot = plot_sp(input_df=input_feature,
                         filllim=filllim,
                         ylab=ylab,
                         xlab=xlab)
        plots.append(plot)
    
    return plots
        
        