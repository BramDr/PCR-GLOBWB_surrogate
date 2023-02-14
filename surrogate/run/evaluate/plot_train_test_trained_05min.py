import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

train_dir = pl.Path("../train/saves/global_05min")
transform_dir = pl.Path("../transform/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")

submasks = [dir.stem for dir in train_dir.iterdir() if dir.is_dir()]
submasks = ["cells_M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    train_submask_dir = pl.Path("{}/{}".format(train_dir, submask))
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    submask_out = pl.Path("{}/{}/train".format(dir_out, submask))

    subsets = [dir.stem for dir in train_submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]
    
    subset = "train_32"
    for subset in subsets:
        print("Working on {}".format(subset), flush = True)
    
        train_subset_dir = pl.Path("{}/{}".format(train_submask_dir, subset))
        transform_subset_dir = pl.Path("{}/{}".format(transform_submask_dir, subset))
        subset_out = pl.Path("{}/{}".format(submask_out, subset))
        
        df_file = pl.Path("{}/output_ts.csv".format(transform_subset_dir))
        true_plot_df = pd.read_csv(df_file, index_col=0)
        true_plot_df["type"] = "Actual"
        
        df_file = pl.Path("{}/output_ts.csv".format(train_subset_dir))
        pred_plot_df = pd.read_csv(df_file, index_col=0)
        pred_plot_df["type"] = "Trained"

        plot_df = pd.concat((true_plot_df, pred_plot_df))
        
        plot_out = pl.Path("{}/output_ts.pdf".format(subset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_ts_seperate(input_df=plot_df, plot_sensitivity=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)
        
        df_file = pl.Path("{}/output_cdf.csv".format(transform_subset_dir))
        true_plot_df = pd.read_csv(df_file, index_col=0)
        true_plot_df["type"] = "Actual"

        df_file = pl.Path("{}/output_cdf.csv".format(train_subset_dir))
        pred_plot_df = pd.read_csv(df_file, index_col=0)
        pred_plot_df["type"] = "Trained"

        plot_df = pd.concat((true_plot_df, pred_plot_df))

        plot_out = pl.Path("{}/output_cdf.pdf".format(subset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_cdf_seperate(input_df=plot_df)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)
        
        df_file = pl.Path("{}/output_sp.csv".format(transform_subset_dir))
        true_plot_df = pd.read_csv(df_file, index_col=0)
        true_plot_df["type"] = "Actual"
        
        df_file = pl.Path("{}/output_sp.csv".format(train_subset_dir))
        pred_plot_df = pd.read_csv(df_file, index_col=0)
        pred_plot_df["type"] = "Trained"

        plot_df = pd.concat((true_plot_df, pred_plot_df))

        plot_out = pl.Path("{}/output_sp.pdf".format(subset_out))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_sp_seperate(input_df=plot_df)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)
        