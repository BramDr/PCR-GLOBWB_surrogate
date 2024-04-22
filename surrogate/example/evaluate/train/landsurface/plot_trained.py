import pathlib as pl
import plotnine as pn
import pandas as pd
import numpy as np

from surrogate.utils.plot import plot_ts_seperate

save_dir = pl.Path("./saves")
setup_dir = pl.Path("../../setup/landsurface/saves/train-test2")
out_dir = pl.Path("./saves")
dataset = "output"

resolution = "30min"

save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

trainsets = [dir.stem for dir in save_resolution_dir.iterdir() if dir.is_dir()]

trainset = trainsets[0]
for trainset in trainsets:
    print("Trainset: {}".format(trainset))
    
    save_trainset_dir = pl.Path("{}/{}".format(save_resolution_dir, trainset))
    setup_trainset_dir = pl.Path("{}/{}".format(setup_resolution_dir, trainset))
    out_trainset_dir = pl.Path("{}/{}".format(out_resolution_dir, trainset))
    
    save_dataset_dir = pl.Path("{}/{}".format(save_trainset_dir, dataset))
    setup_dataset_dir = pl.Path("{}/{}".format(setup_trainset_dir, dataset))
    out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))
    
    df_files = [file for file in setup_dataset_dir.rglob("*_ts.csv")]
    plot_dfs = []
    for df_file in df_files:
        plot_df = pd.read_csv(df_file, index_col=0)
        plot_dfs.append(plot_df)
    actual_plot_df = pd.concat(plot_dfs)
    actual_plot_df["type"] = "actual"
    
    df_files = [file for file in save_dataset_dir.rglob("*_ts.csv")]
    plot_dfs = []
    for df_file in df_files:
        plot_df = pd.read_csv(df_file, index_col=0)
        plot_dfs.append(plot_df)
    predicted_plot_df = pd.concat(plot_dfs)
    predicted_plot_df["type"] = "predicted"
    
    actual_plot_sel = np.isin(actual_plot_df["feature"], predicted_plot_df["feature"].unique())
    actual_plot_df = actual_plot_df.loc[actual_plot_sel]
    plot_df = pd.concat((actual_plot_df, predicted_plot_df))
    plot_df = plot_df.loc[plot_df["date"] <= "2000-12-31"]
    
    plot_out = pl.Path("{}_ts.pdf".format(out_dataset_dir))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_ts_seperate(input_df=plot_df,
                            plot_sensitivity=True)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    
    plot_out = pl.Path("{}_ts-quant.pdf".format(out_dataset_dir))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_ts_seperate(input_df=plot_df,
                            plot_sensitivity=True,
                            quantile_lim=True)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    