import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

setup_dir = pl.Path("../setup/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/prepare/setup")
datasets = ["input", "output"]

dataset = datasets[0]
for dataset in datasets:
    print("Working on {}".format(dataset))
    
    dataset_out = pl.Path("{}/{}".format(dir_out, dataset))

    df_files = [file for file in setup_dir.rglob("{}_*_ts.csv".format(dataset))]
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

    df_files = [file for file in setup_dir.rglob("{}_*_cdf.csv".format(dataset))]
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

    df_files = [file for file in setup_dir.rglob("{}_*_sp.csv".format(dataset))]
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