import pathlib as pl
import plotnine as pn
import pandas as pd

train_dir = pl.Path("../train/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/train")

performance_files = [file for file in train_dir.rglob("*statistics.csv")]

performance_df = pd.DataFrame(columns=["type", "size", "best"])

file = performance_files[0]
for file in performance_files:
    subset = file.parent.name
    
    subset_out = pl.Path("{}/{}".format(dir_out, subset))
    
    performance = pd.read_csv(file)
    performance["index"] = performance.index
    performance = pd.melt(frame=performance,
                          id_vars=["index"],
                          value_vars=["train_loss", "test_loss"],
                          value_name="value",
                          var_name="loss")
    
    plot = pn.ggplot(data=performance,
                    mapping=pn.aes(x="index",
                                   y = "value",
                                   group = "loss"))
    plot += pn.geom_line()
    plot += pn.facet_wrap(facets="~loss", scales="free_y")
    plot += pn.ggtitle(title=subset)
    plot += pn.theme(panel_background=pn.element_blank(),
                    panel_grid_major=pn.element_blank(),
                    panel_grid_minor=pn.element_blank())
    
    plot_out = pl.Path("{}/performance.pdf".format(subset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    pn.save_as_pdf_pages(plots = [plot], filename= plot_out)
    
    if subset == "total":
        continue
    size = int(subset.split("_")[-1])
    
    best = performance["value"].loc[performance["loss"] == "test_loss"].min()
    perforamnce_dict = {"type": [subset], "size": [size], "best": [best]}
    perforamnce_row = pd.DataFrame(perforamnce_dict)
    performance_df = pd.concat((performance_df, perforamnce_row))
    
performance_df = performance_df.reset_index(drop=True)

plot = pn.ggplot(data=performance_df,
                 mapping=pn.aes(x="size",
                                y="best",
                                group = 1))
plot += pn.geom_line()
plot += pn.theme(panel_background=pn.element_blank(),
                panel_grid_major=pn.element_blank(),
                panel_grid_minor=pn.element_blank())

plot_out = pl.Path("{}/train_performance.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
pn.save_as_pdf_pages([plot], filename = plot_out)
