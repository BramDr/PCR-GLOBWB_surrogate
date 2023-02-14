import pathlib as pl
import random

import pandas as pd
import plotnine as pn

save_dir = pl.Path("./saves/global_30min")
dir_out = pl.Path("./saves/global_30min")
seed = 19920223

cells_file = pl.Path("{}/cells.csv".format(save_dir))
cells = pd.read_csv(cells_file, index_col=0)

sizes = [8, 16, 32, 48, 64, 80, 96]
subsets = ["train_{}".format(size) for size in sizes]
sizes += [128]
subsets += ["hyper"]
sizes.reverse()
subsets.reverse()

size = 96
subset = "train_96"
for size, subset in zip(sizes, subsets):
    print("Working on {}".format(subset), flush=True)
    
    subset_out = pl.Path("{}/{}".format(save_dir, subset))
    
    random.seed(a = seed)
    
    samples_len = int(len(cells.index) / size)
    samples = random.sample(population=cells.index.tolist(), k = samples_len)
        
    subset_cells = cells.loc[samples]
    
    cells_out = pl.Path("{}/cells.csv".format(subset_out))
    cells_out.parent.mkdir(parents=True, exist_ok=True)
    subset_cells.to_csv(cells_out)
    
    ggp = pn.ggplot(data = subset_cells, mapping=pn.aes(x = "lon",
                                                        y = "lat"))
    ggp += pn.geom_raster()
    
    plot_out = pl.Path("{}/cells_sp.pdf".format(subset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    pn.save_as_pdf_pages(plots = [ggp], filename=plot_out)
