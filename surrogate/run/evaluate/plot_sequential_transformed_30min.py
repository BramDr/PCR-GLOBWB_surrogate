import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

save_dir = pl.Path("../transform/saves/global_30min/sequential")
dir_out = pl.Path("./saves/global_30min/transform/sequential")
datasets = ["input"]
    
dataset = datasets[0]
for dataset in datasets:
    print("Processing {}".format(dataset))

    df_files = [file for file in save_dir.rglob("*{}_*_ts.csv".format(dataset))]
    plot_dfs = []
    for df_file in df_files:
        plot_df = pd.read_csv(df_file, index_col=0)
        plot_dfs.append(plot_df)
    plot_df = pd.concat(plot_dfs)
    plot_df["type"] = "Transformed"

    plot_out = pl.Path("{}/{}_ts.pdf".format(dir_out, dataset))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    #if not plot_out.exists():
    plots = plot_ts_seperate(input_df=plot_df,
                            plot_sensitivity=True)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)

    df_files = [file for file in save_dir.rglob("*{}_*_cdf.csv".format(dataset))]
    plot_dfs = []
    for df_file in df_files:
        plot_df = pd.read_csv(df_file, index_col=0)
        plot_dfs.append(plot_df)
    plot_df = pd.concat(plot_dfs)
    plot_df["type"] = "Transformed"

    plot_out = pl.Path("{}/{}_cdf.pdf".format(dir_out, dataset))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    #if not plot_out.exists():
    plots = plot_cdf_seperate(input_df=plot_df)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)

    df_files = [file for file in save_dir.rglob("*{}_*_sp.csv".format(dataset))]
    plot_dfs = []
    for df_file in df_files:
        plot_df = pd.read_csv(df_file, index_col=0)
        plot_dfs.append(plot_df)
    plot_df = pd.concat(plot_dfs)
    plot_df["type"] = "Transformed"

    plot_out = pl.Path("{}/{}_sp.pdf".format(dir_out, dataset))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    #if not plot_out.exists():
    plots = plot_sp_seperate(input_df=plot_df)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    