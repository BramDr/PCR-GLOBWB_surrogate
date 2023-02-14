import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

transform_dir = pl.Path("../transform/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["output", "input"]

submasks = [dir.stem for dir in transform_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    setup_submask_dir = pl.Path("{}/{}/setup".format(transform_dir, submask))
    submask_out = pl.Path("{}/{}/transform/setup".format(dir_out, submask))
    
    dataset = datasets[0]
    for dataset in datasets:
        print("Working on {}".format(dataset))

        df_files = [file for file in transform_dir.rglob("{}_*_ts.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Transformed"

        plots = plot_ts_seperate(input_df=plot_df, plot_sensitivity=True)
        plot_out = pl.Path("{}/{}_ts.pdf".format(submask_out, dataset))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)

        df_files = [file for file in transform_dir.rglob("{}_*_cdf.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Transformed"

        plots = plot_cdf_seperate(input_df=plot_df)
        plot_out = pl.Path("{}/{}_cdf.pdf".format(submask_out, dataset))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)

        df_files = [file for file in transform_dir.rglob("{}_*_sp.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Transformed"

        plot_out = pl.Path("{}/{}_sp.pdf".format(submask_out, dataset))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_sp_seperate(input_df=plot_df)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)