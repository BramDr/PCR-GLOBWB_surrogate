import pathlib as pl
import plotnine as pn
import pandas as pd

train_dir = pl.Path("../train/saves/global_30min/total")
dir_out = pl.Path("./saves/global_30min/train/total")

df_file = pl.Path("{}/sensitivity.csv".format(train_dir))
df_file.parent.mkdir(parents=True, exist_ok=True)

plot_df = pd.read_csv(df_file, index_col=0)

out_features = plot_df["out_feature"].unique()

plots = []

out_feature = out_features[0]
for out_feature in out_features:
    plot_df_feature = plot_df.loc[plot_df["out_feature"] == out_feature]
    plot_df_feature["loss"] = plot_df_feature["loss"] / plot_df_feature["loss"].max()
    
    plot = pn.ggplot(data = plot_df_feature,
                    mapping=pn.aes(x = "in_feature",
                                   fill = "sensitivity",
                                   y = "loss"))
    plot += pn.geom_bar(stat="identity")
    plot += pn.facet_wrap(facets="~out_feature")
    plot += pn.theme(panel_background=pn.element_blank(),
                    panel_grid_major=pn.element_blank(),
                    panel_grid_minor=pn.element_blank())
    
    plots.append(plot)
    
plot_out = pl.Path("{}/sensitivity.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
pn.save_as_pdf_pages(plots = plots, filename= plot_out)