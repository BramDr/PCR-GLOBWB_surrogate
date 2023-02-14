import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

correct_dir = pl.Path("../correct/saves/global_30min")
transform_dir = pl.Path("../transform/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/correct")

subsets = ["train_{}".format(size)
           for size in [8, 16, 32, 48, 64, 80, 96]]
subsets.reverse()

subset = "train_96"
for subset in subsets:
    print("Working on {}".format(subset), flush = True)
    
    train_subset_dir = pl.Path("{}/{}".format(correct_dir, subset))
    transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))
    
    df_file = pl.Path("{}/output_ts.csv".format(transform_subset_dir))
    true_plot_df = pd.read_csv(df_file, index_col=0)
    true_plot_df["type"] = "Actual"
        
    df_file = pl.Path("{}/output_ts.csv".format(train_subset_dir))
    pred_plot_df = pd.read_csv(df_file, index_col=0)
    pred_plot_df["type"] = "Corrected"

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
    pred_plot_df["type"] = "Corrected"

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
    pred_plot_df["type"] = "Corrected"

    plot_df = pd.concat((true_plot_df, pred_plot_df))

    plot_out = pl.Path("{}/output_sp.pdf".format(subset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_sp_seperate(input_df=plot_df)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    