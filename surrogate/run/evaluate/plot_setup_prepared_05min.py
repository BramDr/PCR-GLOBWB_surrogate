import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

setup_dir = pl.Path("../setup/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
datasets = ["input", "output"]

submasks = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]
submasks = ["cells_M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    setup_submask_dir = pl.Path("{}/{}".format(setup_dir, submask))
    submask_out = pl.Path("{}/{}/prepare/setup".format(dir_out, submask))
        
    dataset = datasets[0]
    for dataset in datasets:
        print("Working on {}".format(dataset))
        
        dataset_out = pl.Path("{}/{}".format(submask_out, dataset))

        df_files = [file for file in setup_submask_dir.rglob("{}_*_ts.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Setup"

        plots = plot_ts_seperate(input_df=plot_df, plot_sensitivity=True)
        plot_out = pl.Path("{}_ts.pdf".format(dataset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)

        df_files = [file for file in setup_submask_dir.rglob("{}_*_cdf.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Setup"

        plots = plot_cdf_seperate(input_df=plot_df)
        plot_out = pl.Path("{}_cdf.pdf".format(dataset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)

        df_files = [file for file in setup_submask_dir.rglob("{}_*_sp.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "Setup"

        plots = plot_sp_seperate(input_df=plot_df)
        plot_out = pl.Path("{}_sp.pdf".format(dataset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)