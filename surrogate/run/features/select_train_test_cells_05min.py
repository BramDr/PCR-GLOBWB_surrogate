import pathlib as pl
import random

import pandas as pd
import plotnine as pn

save_dir = pl.Path("./saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
seed = 19920223

sizes = [32, 96]
subsets = ["train_{}".format(size) for size in sizes]
sizes.reverse()
subsets.reverse()

submasks = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

for submaks in submasks:
    print("Working on {}".format(submaks))
        
    submask_dir = pl.Path("{}/{}".format(save_dir, submaks))
    submask_out = pl.Path("{}/{}".format(dir_out, submaks))
    
    cells_file = pl.Path("{}/cells.csv".format(submask_dir))
    cells = pd.read_csv(cells_file, index_col=0)

    size = 96
    subset = "train_96"
    for size, subset in zip(sizes, subsets):
        print("Processing {}".format(subset), flush=True)
        
        subset_out = pl.Path("{}/{}".format(submask_out, subset))
        
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
