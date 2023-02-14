import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

transform_dir = pl.Path("../transform/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]

submasks = [dir.stem for dir in transform_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    submask_out = pl.Path("{}/{}/transform".format(dir_out, submask))
    
    subsets = [dir.stem for dir in transform_submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

    subset = "train_96"
    for subset in subsets:
        print("Working on {}".format(subset), flush = True)
        
        transform_subset_dir = pl.Path("{}/{}".format(transform_submask_dir, subset))
        subset_out = pl.Path("{}/{}".format(submask_out, subset))
        
        dataset = datasets[0]
        for dataset in datasets:
            print("Processing {}".format(dataset))

            df_file = pl.Path("{}/{}_ts.csv".format(transform_subset_dir, dataset))
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_df["type"] = "Transformed"

            plot_out = pl.Path("{}/{}_ts.pdf".format(subset_out, dataset))
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plots = plot_ts_seperate(input_df=plot_df,
                                    plot_sensitivity=True)
            pn.save_as_pdf_pages(plots = plots, filename= plot_out)

            df_file = pl.Path("{}/{}_cdf.csv".format(transform_subset_dir, dataset))
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_df["type"] = "Transformed"

            plot_out = pl.Path("{}/{}_cdf.pdf".format(subset_out, dataset))
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plots = plot_cdf_seperate(input_df=plot_df)
            pn.save_as_pdf_pages(plots = plots, filename= plot_out)

            df_file = pl.Path("{}/{}_sp.csv".format(transform_subset_dir, dataset))
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_df["type"] = "Transformed"

            plot_out = pl.Path("{}/{}_sp.pdf".format(subset_out, dataset))
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plots = plot_sp_seperate(input_df=plot_df)
            pn.save_as_pdf_pages(plots = plots, filename= plot_out)